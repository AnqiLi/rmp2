from rmp2.rmpgraph.rmpgraph import RMPGraph
from rmp2.rmpgraph.rmps.rmps import RMP
from rmp2.utils.tf_utils import solve
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

dtype = tf.float64

T = 40.
dt = 0.01

# obstacles

q_0 = tf.convert_to_tensor([[3., -3.]], dtype=dtype)
qd_0 = tf.convert_to_tensor([[-1., 1.]], dtype=dtype)

goals = tf.convert_to_tensor([[-3., 3.]], dtype=dtype)

obstacle_centers = tf.convert_to_tensor([[0., 0.]], dtype=dtype)
obstacle_radii = tf.convert_to_tensor([1.], dtype=dtype)

class CollisionAvoidanceRMP(RMP):
    def __init__(self, epsilon=0.2, alpha=1e-5, eta=0.2, name='collision_avoidance', dtype=tf.float32):

        self.alpha = alpha
        self.eta = eta
        self.epsilon = epsilon

        super(CollisionAvoidanceRMP, self).__init__(name=name, dtype=dtype)

    def rmp_eval_natural(self, x, xd, **features):
        w = tf.where(
            x < 0,
            1e6 * tf.ones_like(x),
            1.0 / x ** 4
        )
        grad_w = tf.where(
            x < 0,
            tf.zeros_like(x),
            -4. / x ** 5
        )
        u = self.epsilon + tf.minimum(xd, tf.zeros_like(xd)) * xd
        g = w * u
        grad_u = 2 * tf.minimum(xd, tf.zeros_like(xd))
        grad_Phi = self.alpha * w * grad_w
        xi = 0.5 * xd ** 2 * u * grad_w
        metric = tf.expand_dims(g + 0.5 * xd * w * grad_u, axis=1)
        metric = tf.clip_by_value(
            metric,
            clip_value_min=-1e5, clip_value_max=1e5
        )
        Bx_dot = self.eta * g * xd
        force = -grad_Phi - xi - Bx_dot
        force = tf.clip_by_value(
            force,
            clip_value_min=-1e10, clip_value_max=1e10
        )
        return metric, force

    def rmp_eval_canonical(self, x, xd, **features):
        metric, force = self.rmp_eval_natural(x, xd, **features)
        accel = solve(metric, force)
        return metric, accel


class GoalAttractorRMP(RMP):
    def __init__(self, w_u=10, w_l=1, sigma=1, alpha=1, eta=2, gain=1, tol=0.005, name='goal_attractor', dtype=tf.float32):
        self.w_u = w_u
        self.w_l = w_l
        self.sigma = sigma
        self.alpha = alpha
        self.eta = eta
        self.gain = gain
        self.tol = tol

        super(GoalAttractorRMP, self).__init__(name=name, dtype=dtype)

    def rmp_eval_natural(self, x, xd, **features):
        batch_size, n_dims = x.shape
        batch_size, n_dims = int(batch_size), int(n_dims)

        x_norm = tf.linalg.norm(x, axis=1)
        x_norm = tf.expand_dims(x_norm, -1)

        beta = tf.exp(-x_norm ** 2 / 2 / (self.sigma ** 2))
        w = (self.w_u - self.w_l) * beta + self.w_l
        s = (1 - tf.exp(-2 * self.alpha * x_norm)) / (1 + tf.exp(-2 * self.alpha * x_norm))

        grad_Phi = s / x_norm * w * x * self.gain
        close_to_goal = x_norm <= tf.ones_like(x_norm) * self.tol
        grad_Phi = tf.where(
            close_to_goal,
            tf.zeros_like(grad_Phi),
            grad_Phi
        )

        Bx_dot = self.eta * w * xd
        grad_w = -beta * (self.w_u - self.w_l) / self.sigma ** 2 * x

        xd_norm = tf.linalg.norm(xd, axis=1)
        S = tf.einsum('bi, bj->bij', xd, xd)
        xi = -0.5 * (xd_norm ** 2 * grad_w - 2 * tf.einsum('bij, bj->bi', S, grad_w))

        metric = tf.eye(n_dims, batch_shape=[batch_size], dtype=self.dtype) * w
        force = -grad_Phi - Bx_dot - xi

        return metric, force

    def rmp_eval_canonical(self, x, xd, **features):
        metric, force = self.rmp_eval_natural(x, xd, **features)
        accel = solve(metric, force)
        return metric, accel


def pdist2(x, y, epsilon=1e-4):
    """
    distance between 2 vectors
    x: [n, d]
    y: [m, d]
    epsilon: scalar for numerical stability
    return: dist: [n, m], dist(i, j) = |x[i,:] - y[j,:]|
    """
    n, d = x.shape
    m = y.shape[1]
    dtype = x.dtype
    epsilon = tf.convert_to_tensor(epsilon, dtype=dtype)

    x2 = tf.einsum('ij, jk->ik', x ** 2, tf.ones((d, m), dtype=dtype))
    y2 = tf.einsum('ij, kj->ik', tf.ones((n, d), dtype=dtype), y ** 2)
    xy = tf.einsum('ij, kj->ik', x, y)

    dist_sq = x2 + y2 - 2 * xy
    return tf.sqrt(dist_sq + epsilon) - tf.sqrt(epsilon)


class PointMassRMPGraph(RMPGraph):
    """
    RMP graph for point mass robot in 2d
    """

    def __init__(self, rmp_type='canonical', timed=False, offset=0.001, dtype=tf.float32, name='point_mass'):

        # goal attractors
        rmps = [
            GoalAttractorRMP(dtype=dtype),
            CollisionAvoidanceRMP(dtype=dtype)
        ]

        super().__init__(rmps, rmp_type, timed, offset, dtype, name)

    def forward_mapping(self, q):

        # delta vector between robot and goals
        _, n_dims = q.shape
        delta = tf.expand_dims(q, 1) - tf.expand_dims(goals, 0)
        goal_attractor_x = tf.reshape(delta, (-1, n_dims))

        # distance between robot and obstacles

        center_dists = pdist2(q, obstacle_centers)
        obstacle_dist = center_dists - obstacle_radii
        collision_avoidance_x = tf.reshape(obstacle_dist, (-1, 1))

        return [goal_attractor_x, collision_avoidance_x]


graph = PointMassRMPGraph(dtype=dtype)


traj = []
q = q_0
qd = qd_0

for t in tqdm(np.arange(0, T, dt)):
    qdd = graph(q, qd)
    q = q + qd * dt
    qd = qd + qdd * dt
    traj.append(q.numpy().copy())

traj = np.concatenate(traj, axis=0)


plt.plot(traj[:, 0], traj[:, 1])

print(traj.shape)

for goal in goals:
    goal_np = goal.numpy()
    plt.plot(goal_np[0], goal_np[1], 'go')

for (obs_c, obs_r) in zip(obstacle_centers, obstacle_radii):
    obs_c_np = obs_c.numpy()
    obs_r_np = obs_r.numpy()
    circle = plt.Circle((obs_c_np[0], obs_c_np[1]), obs_r_np, color='k', fill=False)
    plt.gca().add_artist(circle)

plt.axis([-5, 5, -5, 5])
plt.gca().set_aspect('equal', 'box')
plt.savefig('point.png')
