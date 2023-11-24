import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

def plot_trajectory(traj):
    """
    plot the 3d trajectory
    """
    traj = traj.reshape((traj.shape[0], traj.shape[1]))
    print(traj, traj.shape)
    x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
    # print(x, y, z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, marker='x', color='r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory Plot')

    # Show the plot
    plt.show()


def hat(x):
    if not x.ndim == 1:
        x = x.reshape(3, )
    return np.array(
        [
            [0, -x[2], x[1]],
            [x[2], 0, -x[0]],
            [-x[1], x[0], 0]
        ]
    )


def q_cross(q):
    """
    define the cross operator for state vector q = [p.T, r1.T, r2.T, r3.T]
    p is the position vector, ri is the ith column of the rotation matrix of the system
    """
    R = q[3:12].reshape((3, 3))
    r1, r2, r3 = R[:, 0], R[:, 1], R[:, 2]
    return np.block(
        [
            [R, np.zeros((3, 3))],
            [np.zeros((3, 3)), hat(r1)],
            [np.zeros((3, 3)), hat(r2)],
            [np.zeros((3, 3)), hat(r3)]
        ]
    )


def generalized_velocity(v, omega):
    """
    concatenate linear and angular velocity to get generalized velocity of system
    """
    zeta = np.vstack(
        (
            v, omega
        )
    ).reshape((6, 1))
    assert zeta.shape == (6, 1), "generalized velocity is not of shape 6 X 1"
    return zeta


def interconnectivity_matrix(q, pv, pw):
    """
    define the interconnectivity matrix J(q, p) where p is the momenta of the body
    """
    qx = q_cross(q)
    pv_hat, pw_hat = hat(pv), hat(pw)
    px = np.block(
        [
            [np.zeros((3, 3)), pv_hat],
            [pv_hat, pw_hat]
        ]
    )
    return np.block(
        [
            [np.zeros((12, 12)), qx],
            [-qx.T, px]
        ]
    )


class Hamiltonian:
    """
    define the hamiltonian equation and system properties
    """
    def __init__(self, mass: float = 5, Jxx: float = 0.2, Jyy: float = 0.2, Jzz: float = 0.4):
        """
        initialize physical properties like mass, inertia
        """
        self.mass = mass * np.eye(3)
        self.Inertia = np.diag([Jxx, Jyy, Jzz])
        self.g = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        self.x = np.zeros((3, 1))
        self.v = np.zeros_like(self.x)
        self.omega = np.zeros_like(self.x)
        self.R = np.eye(3)

    def dynamics(self, t, y):
        """
        define the dynamics equations of the system
        """
        u = y[-4:].reshape((4, 1))
        q = np.vstack(
            (self.x, self.R.flatten().reshape((-1, 1)))
        )
        pv = self.mass @ self.v
        pw = self.Inertia @ self.omega
        dHdp = generalized_velocity(self.v, self.omega)
        dHdq = np.array([0, 0, 9.8 * self.mass[0, 0], 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((12, 1))
        J = interconnectivity_matrix(q=q, pv=pv, pw=pw)
        q_p_dot = J @ np.vstack(
            (
                dHdq,
                dHdp
            )
        ).reshape((J.shape[0], -1))
        q_dot = q_p_dot[:12]
        p_dot = q_p_dot[12:] + self.g @ u
        pv_dot = p_dot[:3]
        pw_dot = p_dot[3:]
        # print(f"pv_dot : {pv_dot} | pw_dot : {pw_dot}")
        v_dot = np.linalg.inv(self.mass) @ pv_dot
        omega_dot = np.linalg.inv(self.Inertia) @ pw_dot
        return np.hstack(
            (q_dot.flatten(), v_dot.flatten(), omega_dot.flatten(), np.zeros(4, ))
        )

    def evolve_states(self, u):
        """
        given the derivatives of the state variables, find the next state by integrating
        """
        dt = 0.5
        if u.shape != (4, 1):
            u = u.reshape((4, 1))
        t_span = (0, dt)
        y0 = np.hstack(
            (self.x.flatten(), self.R.flatten(), self.v.flatten(), self.omega.flatten(), u.flatten())
        )
        sol = solve_ivp(self.dynamics, t_span, y0, t_eval=t_span)
        y = sol.y[:, -1]
        self.x = y[:3].reshape((3, 1))
        self.R = y[3:12].reshape((3, 3))
        self.v = y[12:15].reshape((3, 1))
        self.omega = y[15:18].reshape((3, 1))

    def run(self, control_actions, x0, R0, v0, omega0):
        """
        run the simulation for given control actions
        """
        assert control_actions.ndim == 3 and control_actions.shape[1] == 4
        print(f"thrusts : {control_actions[:, 0]}")
        self.x = x0
        self.v = v0
        self.omega = omega0
        self.R = R0
        print(f"x : {x0} | v : {v0} | omega : {omega0} | R : {R0}")
        trajectory = []
        for u in tqdm(control_actions):
            self.evolve_states(u)
            print(f"x : {self.x} | v : {self.v} | R : {self.R}")
            trajectory.append(
                self.x
            )
        trajectory = np.array(trajectory)
        return self.x - x0


if __name__ == '__main__':
    hnode = Hamiltonian()
    controls = np.zeros((2, 4, 1))
    controls[:, 0, :] = 9.8 * 9
    x0 = np.zeros((3, 1))
    v0, omega0 = np.zeros_like(x0), np.zeros_like(x0)
    R0 = np.eye(3)
    trajectory = hnode.run(control_actions=controls, x0=x0, R0=R0, v0=v0, omega0=omega0)
    plot_trajectory(traj=trajectory)






