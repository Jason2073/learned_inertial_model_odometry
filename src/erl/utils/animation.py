import time 
import numpy as np 
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation
from matplotlib.patches import Circle, FancyArrow
from matplotlib.animation import PillowWriter
from scipy.interpolate import make_interp_spline, BSpline
from matplotlib.animation import FFMpegWriter


class Animation:
    def __init__(self, trajectory = None, target_trajectory = None, sim_freq = 100, show=True, label=None):
        self.plt_trajectory = None 
        self.plt_target_trajectory = None 
        self.plt_ax = None 
        self.plt_robot = None
        self.plt_line = None 
        self.goal = None 
        self.goal_heading = None 
        self.plt_initial_pose = None 
        self.plt_initial_heading = None 
        self.plt_steps = None 
        self.fig, self.plt_ax = plt.subplots()
        self.sim_freq = sim_freq
        
        if trajectory is None:
            self.trajectory = None 
        else:
            self.trajectory = trajectory
        if target_trajectory is None:
            self.target_trajectory = None 
        else:
            self.target_trajectory = target_trajectory
        if label is None:
            self.label = None 
        else:
            self.label = label 

        # Set up formatting for the movie files
        self.Writer = FFMpegWriter(fps=4*int(1/sim_freq), metadata=dict(artist='Me'), bitrate=500)

    def set(self, trajectory, target_trajectory, label):
        self.trajectory = trajectory
        self.target_trajectory = target_trajectory
        self.label = label 

    def animate(self, show=True):
        '''
        single_trajectory       : (N, 3)-shaped numpy array 
        single_target_trajectory: (N, 3)-shaped numpy array 
        '''
        samples = self.trajectory.shape[0]
        position = self.trajectory[:, 0:3]
        orientation = self.trajectory[:, 3:12]
        linear_velocity =self.trajectory[:, 12:15]
        angular_velocity = self.trajectory[:, 15:18]
        control = self.trajectory[:, 18:20]

        target_position = self.target_trajectory[:, 0:3]
        target_orientation = self.target_trajectory[:, 3:12]
        target_linear_velocity =self.target_trajectory[:, 12:15]
        target_angular_velocity = self.target_trajectory[:, 15:18]

        x = position[:, 0]
        y = position[:, 1]
        yaw = np.zeros(samples)
        target_yaw = np.zeros(samples)
        for i in range(samples):
            R = Rotation.from_matrix(orientation[i, :].reshape(3,3))
            euler = R.as_euler('xyz')
            yaw[i] = euler[2] * 180.0/np.pi

            R = Rotation.from_matrix(target_orientation[i, :].reshape(3,3))
            euler = R.as_euler('xyz')
            target_yaw[i] = euler[2] * 180.0/np.pi

        self.plt_trajectory = np.concatenate((position[:, 0, np.newaxis], position[:, 1, np.newaxis], yaw[:, np.newaxis]), axis=1)
        self.plt_target_trajectory = np.concatenate((target_position[:, 0, np.newaxis], target_position[:, 1, np.newaxis], target_yaw[:, np.newaxis]), axis=1)

        # Number of steps for the animation
        self.plt_steps = self.plt_trajectory.shape[0]

        # Initialize the robot and heading arrow
        self.plt_robot = Circle((self.plt_trajectory[0,0], self.plt_trajectory[0,1]), 0.5, fill=True)
        self.plt_ax.add_patch(self.plt_robot)

        # Initialize the trajectory line
        self.plt_line, = plt.plot([], [], 'b-')

        # Mark the goal position as a cross ('x') and add a heading arrow
        self.goal, = plt.plot(self.plt_target_trajectory[-1, 0], self.plt_target_trajectory[-1, 1], 'rx')
        self.goal_heading = FancyArrow(self.plt_target_trajectory[-1, 0], self.plt_target_trajectory[-1, 1], np.cos(self.plt_target_trajectory[-1, 2]*np.pi/180), np.sin(self.plt_target_trajectory[-1, 2]*np.pi/180), color='red', width=0.2)
        self.plt_ax.add_patch(self.goal_heading)

        # Initialize the initial pose of the robot and its heading as a faded shape
        self.plt_initial_pose = Circle((self.plt_trajectory[0, 0], self.plt_trajectory[0, 1]), 0.5, fill=True, alpha=0.3)
        self.plt_initial_heading = FancyArrow(self.plt_trajectory[0, 0], self.plt_trajectory[0, 1], np.cos(self.plt_trajectory[0, 2]*np.pi/180), np.sin(self.plt_trajectory[0, 2]*np.pi/180), color='gray', width=0.2, alpha=0.3)
        self.plt_ax.add_patch(self.plt_initial_pose)
        self.plt_ax.add_patch(self.plt_initial_heading)

        ani = FuncAnimation(self.fig, self.update, frames=range(self.plt_steps), init_func=self.plt_init, blit=True, interval=1/self.sim_freq, repeat=False)
        if show:
            plt.show()
            video_title = f'{self.label}'
            # ani.save(video_title, writer=self.Writer)
        
        self.plot()


    def plt_init(self):
        self.plt_ax.set_xlim(-5, 5)
        self.plt_ax.set_ylim(-5, 5)
        self.plt_ax.grid(True)  # Display a grid
        return self.plt_robot, self.plt_line, self.goal, self.goal_heading, self.plt_initial_pose, self.plt_initial_heading
    
    def update(self, i):
        '''
        # Function to update the plot objects for matplotlib.animation
        '''
        self.plt_robot.center = self.plt_trajectory[i,0], self.plt_trajectory[i,1]
    
        # remove all arrows
        for arrow in self.plt_ax.patches:
            if isinstance(arrow, FancyArrow) and arrow is not self.goal_heading and arrow is not self.plt_initial_heading:
                arrow.remove()
        
        # new arrow indicating the heading
        heading = FancyArrow(self.plt_trajectory[i,0], self.plt_trajectory[i,1], np.cos(self.plt_trajectory[i,2]*np.pi/180), np.sin(self.plt_trajectory[i,2]*np.pi/180), color='blue', width=0.2)
        self.plt_ax.add_patch(heading)

         # update the trajectory line
        self.plt_line.set_data(self.plt_trajectory[:i+1, 0], self.plt_trajectory[:i+1, 1])

        # # Save last frame 
        # print(f'({i}) out of {self.plt_steps}')
        # if (i == self.plt_steps - 1):
        #     plt.savefig(self.label)
        
        return self.plt_robot, heading, self.plt_line, self.goal, self.goal_heading, self.plt_initial_pose, self.plt_initial_heading
    
    def plot(self):
        samples = self.trajectory.shape[0]

        # Extracting data segments
        xyz = self.trajectory[:, 0:3]
        orientation_matrices = self.trajectory[:, 3:12].reshape(samples, 3, 3)
        linear_velocities = self.trajectory[:, 12:15]
        angular_velocities = self.trajectory[:, 15:18]

        # Convert orientation matrices to Euler angles
        euler_angles = np.array([Rotation.from_matrix(matrix).as_euler('xyz') for matrix in orientation_matrices])

        # Plotting
        fig, axs = plt.subplots(4, 1, figsize=(8, 12))

        # Plot xyz position
        axs[0].plot(xyz)
        axs[0].legend(['X', 'Y', 'Z'])
        axs[0].set_title('XYZ Position')
        axs[0].set_xlabel('Samples')
        axs[0].set_ylabel('Position')
        axs[0].grid(True)

        # Plot Euler angles
        axs[1].plot(euler_angles)
        axs[1].legend(['Roll', 'Pitch', 'Yaw'])
        axs[1].set_title('Euler Angles')
        axs[1].set_xlabel('Samples')
        axs[1].set_ylabel('Angle (rad)')
        axs[1].grid(True)

        # Plot linear velocities
        axs[2].plot(linear_velocities)
        axs[2].legend(['Vx', 'Vy', 'Vz'])
        axs[2].set_title('Linear Velocities')
        axs[2].set_xlabel('Samples')
        axs[2].set_ylabel('Velocity')
        axs[2].grid(True)

        # Plot angular velocities
        axs[3].plot(angular_velocities)
        axs[3].legend(['ωx', 'ωy', 'ωz'])
        axs[3].set_title('Angular Velocities')
        axs[3].set_xlabel('Samples')
        axs[3].set_ylabel('Angular Velocity')
        axs[3].grid(True)

        plt.tight_layout()
        plt.show()


                