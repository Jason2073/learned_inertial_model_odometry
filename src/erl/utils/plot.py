import numpy as np  
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation, ArtistAnimation
from matplotlib.patches import Circle, FancyArrow, Rectangle
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from scipy.interpolate import make_interp_spline, BSpline

from matplotlib.animation import FFMpegWriter

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['text.usetex'] = True

def plot(trajectory, target_trajectory=None, action_sequences=None, sim_freq=100,
         save_path="./img/differential_drive_trajectory.pdf", traj_label="output", target_label="reference"):
    '''
    trajectory needs to be a [N, 4] with the 4 being (x,y,theta,vx,wz)
    '''
    t_eval = np.arange(0.0, trajectory.shape[0]/sim_freq, 1/sim_freq)[:trajectory.shape[0]]
    t_eval2 = np.arange(0.0, target_trajectory.shape[0]/sim_freq, 1/sim_freq)[:target_trajectory.shape[0]]
    pdf = PdfPages(save_path)
    figsize = (24, 18)
    fontsize = 24
    fontsize_ticks = 32
    line_width = 1
    fig, axs = plt.subplots(3, 2, figsize=figsize)
    axs[0,0].plot(t_eval, trajectory[:,0], 'b', linewidth=line_width, label=traj_label)
    if target_trajectory is not None:
        axs[0,0].plot(t_eval2, target_trajectory[:,0], 'k--', linewidth=line_width, label=target_label)
    axs[0,0].legend(loc="upper right")
    axs[0,0].set(ylabel=r'$x(t)$')
    axs[0,0].set(title='Position x')
    axs[0,0].grid()
    axs[0,0].set_xlabel("time (seconds)")
    axs[1,0].plot(t_eval, trajectory[:,1], 'b', linewidth=line_width, label=traj_label)
    if target_trajectory is not None:
        axs[1,0].plot(t_eval2, target_trajectory[:,1], 'k--', linewidth=line_width, label=target_label)
    axs[1,0].legend(loc="upper right")
    axs[1,0].set(ylabel=r'$y(t)$')
    axs[1,0].set(title='Position y')
    axs[1,0].grid()
    axs[1,0].set_xlabel("time (seconds)")
    axs[2,0].plot(t_eval, trajectory[:,2], 'b', linewidth=line_width, label=traj_label)
    if target_trajectory is not None:
        axs[2,0].plot(t_eval2, target_trajectory[:,2], 'k--', linewidth=line_width, label=target_label)
    axs[2,0].legend(loc="upper right")
    axs[2,0].set(ylabel=r'$\theta(t)$ (degree)')
    axs[2,0].set(title='Heading Angle')
    axs[2,0].grid()
    axs[2,0].set_xlabel("time (seconds)")
    axs[0,1].scatter(trajectory[:,0], trajectory[:,1], color='b', s=line_width, label=traj_label)
    if target_trajectory is not None:
        axs[0,1].scatter(target_trajectory[:,0], target_trajectory[:,1], color='r', s=line_width, label=target_label)
    axs[0,1].set(title='2D Trajectory')
    axs[0,1].legend(loc="upper right")
    axs[0,1].grid()
    axs[0,1].set_xlabel("x-axis")
    if target_trajectory is not None:
        minx = np.floor(min(min(trajectory[:,0]), min(target_trajectory[:,0])) + .1)
        maxx = np.ceil(max(max(trajectory[:,0]), max(target_trajectory[:,0])) + .1)
        miny = np.floor(min(min(trajectory[:,1]), min(target_trajectory[:,1])) + .1)
        maxy = np.ceil(max(max(trajectory[:,1]), max(target_trajectory[:,1])) + .1)
    else:
        minx = np.floor(min(trajectory[:, 0]) + .1)
        maxx = np.ceil(max(trajectory[:, 0]) + .1)
        miny = np.floor(min(trajectory[:, 1]) + .1)
        maxy = np.ceil(max(trajectory[:, 1]) + .1)

    minx = np.floor(min(trajectory[:, 0]) + .1)
    maxx = np.ceil(max(trajectory[:, 0]) + .1)
    miny = np.floor(min(trajectory[:, 1]) + .1)
    maxy = np.ceil(max(trajectory[:, 1]) + .1)
    axs[0,1].set_xlim([minx, maxx])
    axs[0,1].set_ylabel("y-axis")
    axs[0,1].set_ylim([miny, maxy])

    start, end = axs[0,1].get_xlim()
    axs[0,1].xaxis.set_ticks(np.arange(start, end, 1))
    start, end = axs[0,1].get_ylim()
    axs[0,1].yaxis.set_ticks(np.arange(start, end, 1))
    axs[0,1].set_aspect('equal', adjustable='box')
    axs[1,1].plot(t_eval, trajectory[:,3], 'b', linewidth=line_width, label=traj_label)

    if target_trajectory is not None:
        axs[1,1].plot(t_eval2, target_trajectory[:,3], 'k--', linewidth=line_width, label=target_label)
    axs[1,1].legend(loc="upper right")
    axs[1,1].set(ylabel=r'$v_x(t)$')
    axs[1,1].set(title='Linear Velocity')
    axs[1,1].grid()
    axs[1,1].set_xlabel("time (seconds)")
    axs[2,1].plot(t_eval, trajectory[:,4], 'b', linewidth=line_width, label=traj_label)

    if target_trajectory is not None:
        axs[2,1].plot(t_eval2, target_trajectory[:,4], 'k--', linewidth=line_width, label=target_label)
    axs[2,1].legend(loc="upper right")
    axs[2,1].set(ylabel=r'$\omega_z(t)$')
    axs[2,1].set(title='Angular Velocity')
    axs[2,1].grid()
    axs[2,1].set_xlabel("time (seconds)")
    plt.savefig(pdf, format='pdf')
    plt.show()
    pdf.close()



def plot_comparison(trajectory, target_trajectory, action_sequences, sim_freq):
    '''
    trajectory: (num_controllers, num_trajectories, samples, 6)-shaped array 
    target_trajectory: (num_trajectories, samples, 6)-shaped array 
    '''
    # PDF 1 
    u_max = 10
    num_controllers = trajectory.shape[0]
    num_trajectories = trajectory.shape[1]
    t_eval = np.arange(0.0, trajectory.shape[2]/sim_freq, 1/sim_freq)
    pdf = PdfPages("./img/trajectories.pdf")
    # plt.rcParams['text.usetex'] = True


    if (num_controllers == 1):
        for j in range(num_trajectories):
            # Create the figure
            fig = plt.figure(figsize=(8, 11))

            text = f'x0 = {np.around(trajectory[0, j, 0, 0], decimals=2)}, y0 = {np.around(trajectory[0, j, 0, 1], decimals=2)}, theta0 = {np.around(trajectory[0, j, 0, 2], decimals=2)}'
            fig.suptitle(text, fontsize=16)

            # Create the grid
            gs = gridspec.GridSpec(5, 1)

            # First column with 3 rows
            for i in range(3):
                ax = plt.subplot(gs[i, 0])
                ax.plot(t_eval, trajectory[0, j, :, i], label='Energy-Based')
                ax.plot(t_eval, target_trajectory[j, :, i], 'k--', alpha=0.7, label='Reference')
                ax.legend(loc="upper right")
                ax.grid()
                ax.set_xlabel("time (seconds)")
                if i == 0:
                    ax.set_ylabel("x")
                elif i == 1:
                    ax.set_ylabel("y")
                elif i == 2:
                    ax.set_ylabel("theta")
            for i in range(2):
                ax = plt.subplot(gs[3+i, 0])
                ax.plot(t_eval, action_sequences[0, j, :, i], label='Energy-Based')
                ax.legend(loc="upper right")
                ax.grid()
                ax.set_xlabel("time (seconds)")
                if i == 0:
                    ax.hlines(u_max, t_eval[0], t_eval[-1], linestyles='dashed', alpha=0.7)
                    ax.hlines(-u_max, t_eval[0], t_eval[-1], linestyles='dashed', alpha=0.7)
                    ax.set_ylabel("body force")
                    ax.set_ylim([-1.2*u_max, 1.2*u_max])
                elif i == 1:
                    ax.hlines(0.1*u_max, t_eval[0], t_eval[-1], linestyles='dashed', alpha=0.7)
                    ax.hlines(-0.1*u_max, t_eval[0], t_eval[-1], linestyles='dashed', alpha=0.7)
                    ax.set_ylabel("body torque")
                    ax.set_ylim([-0.12*u_max, 0.12*u_max])

            # Save the figure to the PDF
            pdf.savefig(fig)

        plt.close('all')
        pdf.close()

def plot_in_single_figure(trajectory, target_trajectory, sim_freq):
    '''
    trajectory: (num_controllers, num_trajectories, samples, 6)-shaped array 
    target_trajectory: (num_trajectories, samples, 6)-shaped array 
    '''
    # PDF 1 
    num_controllers = trajectory.shape[0]
    num_trajectories = trajectory.shape[1]
    t_eval = np.arange(0.0, trajectory.shape[2]/sim_freq, 1/sim_freq)
    pdf = PdfPages("./img/plot_in_single_figure.pdf")

    # Create the figure
    fig = plt.figure(figsize=(8, 11))

    text = f'Number of trajectories = {num_trajectories}'
    fig.suptitle(text, fontsize=16)

    # Create the grid
    gs = gridspec.GridSpec(5, 1)

    # Create a colormap to generate colors
    color_map = cm.get_cmap('viridis', num_trajectories)


    if (num_controllers == 1):
        for j in range(num_trajectories):
            # Select a color based on the current trajectory number
            color = color_map(j)

            # First column with 3 rows
            for i in range(5):
                ax = plt.subplot(gs[i, 0])
                # ax.plot(t_eval, trajectory[0, j, :, i], color=color)
                if (i == 4):
                    ax.plot(t_eval, trajectory[0, j, :, i+1], color=color)
                else:
                    ax.plot(t_eval, trajectory[0, j, :, i], color=color)
                ax.plot(t_eval, target_trajectory[j, :, i], 'k--', alpha=0.7)
                # ax.legend(loc="upper right")
                ax.grid()
                ax.set_xlabel("time (seconds)")
                if i == 0:
                    ax.set_ylabel("x")
                elif i == 1:
                    ax.set_ylabel("y")
                elif i == 2:
                    ax.set_ylabel("theta")
                elif i == 3:
                    ax.set_ylabel("vx")
                elif i == 4:
                    ax.set_ylabel("wz")
    
        # Save the figure to the PDF
        pdf.savefig(fig)
        plt.close('all')
        pdf.close()

def plotLyapunovFunction(Lyapunov, sim_freq):
    t_eval = np.arange(0.0, Lyapunov.shape[0]/sim_freq, 1/sim_freq)
    pdf = PdfPages("./img/differential_drive_Lyapunov.pdf")
    figsize = (24, 18)
    fontsize = 24
    fontsize_ticks = 32
    line_width = 1
    fig, axs = plt.subplots(3, 1, figsize=figsize)
    axs[0].plot(t_eval, Lyapunov[:,0], 'b', linewidth=line_width, label='1st term')
    axs[0].legend(loc="upper right")
    axs[0].set(ylabel=r'$V_1(t)$')
    axs[0].set(title='V1')
    axs[0].grid()
    axs[0].set_xlabel("time (seconds)")
    axs[1].plot(t_eval, Lyapunov[:,1], 'r', linewidth=line_width, label='2nd term')
    axs[1].legend(loc="upper right")
    axs[1].set(ylabel=r'$V_2(t)$')
    axs[1].set(title='V2')
    axs[1].grid()
    axs[1].set_xlabel("time (seconds)")
    axs[2].plot(t_eval, Lyapunov[:,2], 'k', linewidth=line_width, label='3rd term')
    axs[2].legend(loc="upper right")
    axs[2].set(ylabel=r'$V_3(t)$')
    axs[2].set(title='V3')
    axs[2].grid()
    axs[2].set_xlabel("time (seconds)")
    plt.savefig(pdf, format='pdf') 
    pdf.close()

def plot_2D_controllers_comparison(trajectory, target_trajectory, action_sequences, sim_freq):
    '''
    Input (trajectory): (ctrl_num, traj_num, samples, 6)-shaped numpy array 
    Input (target_trajectory): (traj_num, samples, 6)-shaped numpy array
    '''
    ctrl_num = trajectory.shape[0]
    traj_num = trajectory.shape[1]
    samples = trajectory.shape[2]

    # PDF 
    width = 8 
    height = 8
    font_size = 16
    pdf = PdfPages("./img/controllers.pdf")

    # Create a colormap to generate colors
    radius = 0.5 
    transparent = 0.6
    scaling_factor = 0.5

    colors = ['#6dbce9', '#f3dda2', '#7ba848', '#db6e6a', '#dfa16c']

    # Figure for each trajectory 
    for controller in range(ctrl_num):
        # Create the figure
        fig = plt.figure(figsize=(width, height))
        # text = f'Controller {controller}'
        # fig.suptitle(text, fontsize=font_size)

        # Create the grid
        gs = gridspec.GridSpec(1, 1)

        for traj in range(traj_num):
            color = colors[traj % len(colors)]
            x = trajectory[controller, traj, :, 0]
            y = trajectory[controller, traj, :, 1]
            theta = trajectory[controller, traj, :, 2]
            t = np.linspace(0, 1, samples, endpoint=True)
            spl = make_interp_spline(t, np.c_[x, y], k=3)
            smooth_trajectory = spl(t)
            ax = plt.subplot(gs[0, 0])
            ax.plot(smooth_trajectory[:, 0], smooth_trajectory[:, 1], color=color)
            
            initial_pose = Circle((x[0], y[0]), radius, facecolor=color, fill=True, alpha=transparent, edgecolor='black', linewidth=0.5)
            initial_heading = FancyArrow(x[0], y[0], np.cos(theta[0]*np.pi/180)*scaling_factor, np.sin(theta[0]*np.pi/180)*scaling_factor, 
                                         color='black', width=0.1, alpha=transparent, head_length=0.5)
            
            final_pose = Circle((x[-1], y[-1]), radius, facecolor=color, fill=True, alpha=1, edgecolor='black', linewidth=0.5)
            final_heading = FancyArrow(x[-1], y[-1], np.cos(theta[-1]*np.pi/180)*scaling_factor, np.sin(theta[-1]*np.pi/180)*scaling_factor, 
                                         color='black', width=0.1, alpha=1, head_length=0.5)
            
            # Specify the range and number of ticks for x and y axes
            x_ticks = np.linspace(start=-5, stop=5, num=11)  # 11 ticks from -5 to 5
            y_ticks = np.linspace(start=-5, stop=5, num=11)  # 11 ticks from -5 to 5

            

            
            ax.add_patch(initial_pose)
            ax.add_patch(initial_heading)
            ax.add_patch(final_pose)
            ax.add_patch(final_heading)

            # Set the ticks on the x and y axes
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)

            # Set the x and y spines to be black and with the specified thickness
            border_thickness = 7
            for spine in ['top', 'bottom', 'left', 'right']:
                ax.spines[spine].set_linewidth(border_thickness)
                ax.spines[spine].set_edgecolor('black')

            ax.grid(True)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_xlim([-5, 5])
            ax.set_ylim([-5, 5])

        # Save the figure to the PDF
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
    plt.close('all')
    pdf.close()

def plot_trajectory(trajectory, target_trajectory, action_sequences, sim_freq):
    '''
    Input (trajectory): (ctrl_num, traj_num, samples, 6)-shaped numpy array 
    Input (target_trajectory): (traj_num, samples, 6)-shaped numpy array
    '''
    ctrl_num = trajectory.shape[0]
    traj_num = trajectory.shape[1]
    samples = trajectory.shape[2]

    # PDF 
    width = 12 
    height = 8
    font_size = 16
    
    ######################################################
    # Pose 
    pdf = PdfPages("./img/pose.pdf")
    colors = ['#6dbce9', '#f3dda2', '#7ba848', '#db6e6a', '#dfa16c']

    # Figure for each trajectory 
    for controller in range(ctrl_num):
        # Create the figure
        fig = plt.figure(figsize=(width, height))

        # Create the grid
        gs = gridspec.GridSpec(3, 1)

        for traj in range(traj_num):
            color = colors[traj % len(colors)]
            x = trajectory[controller, traj, :, 0]
            y = trajectory[controller, traj, :, 1]
            theta = trajectory[controller, traj, :, 2]
            vx = trajectory[controller, traj, :, 3]
            vy = trajectory[controller, traj, :, 4]
            w = trajectory[controller, traj, :, 5]
            t = np.linspace(0, samples / (sim_freq) , samples, endpoint=True)
            ax = plt.subplot(gs[0, 0])
            ax.plot(t, x, color=color)
            ax.set_ylabel("x")
            ax.grid(True)
            ax = plt.subplot(gs[1, 0])
            ax.plot(t, y, color=color)
            ax.set_ylabel("y")
            ax.grid(True)
            ax = plt.subplot(gs[2, 0])
            ax.plot(t, theta, color=color)
            ax.set_ylabel("theta")
            ax.grid(True)
            ax.set_xlabel("Time")


        # Save the figure to the PDF
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
    plt.close('all')
    pdf.close()

    ######################################################
    # Twist 
    pdf = PdfPages("./img/twist.pdf")
    colors = ['#6dbce9', '#f3dda2', '#7ba848', '#db6e6a', '#dfa16c']

    # Figure for each trajectory 
    for controller in range(ctrl_num):
        # Create the figure
        fig = plt.figure(figsize=(width, height))

        # Create the grid
        gs = gridspec.GridSpec(3, 1)

        for traj in range(traj_num):
            color = colors[traj % len(colors)]
            x = trajectory[controller, traj, :, 0]
            y = trajectory[controller, traj, :, 1]
            theta = trajectory[controller, traj, :, 2]
            vx = trajectory[controller, traj, :, 3]
            vy = trajectory[controller, traj, :, 4]
            w = trajectory[controller, traj, :, 5]
            t = np.linspace(0, samples / (sim_freq) , samples, endpoint=True)
            ax = plt.subplot(gs[0, 0])
            ax.plot(t, vx, color=color)
            ax.set_ylabel("vx")
            ax.grid(True)
            ax = plt.subplot(gs[1, 0])
            ax.plot(t, vy, color=color)
            ax.set_ylabel("vy")
            ax.grid(True)
            ax = plt.subplot(gs[2, 0])
            ax.plot(t, w, color=color)
            ax.set_ylabel("w")
            ax.grid(True)
            ax.set_xlabel("Time")


        # Save the figure to the PDF
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
    plt.close('all')
    pdf.close()



int_fig = None
int_ax = None
plt_robot = None
goal_heading = None
rpe_arrow = None
plt_initial_heading = None
int_trajx = []
int_trajy = []
plt_line = None
bar_ax = None
# int_text = None
rects = None
scale = .5
writer = None
plt.rcParams.update({'font.size': 20})
def stop_writer():
    global writer

def init_interactive_plot(params, initial, goalpos):

    global int_fig, int_ax, plt_robot, goal_heading, plt_initial_heading, plt_line, int_trajx, int_trajy, int_text, \
        rpe_arrow, bar_ax, rects, writer
    # Number of steps for the animation
    # Create a new figure for the plot
    writer = FFMpegWriter(fps=20)


    plt.ion()

    int_trajx = []
    int_trajy = []
    if int_fig is None or int_ax is None:
        int_fig, (int_ax, bar_ax) = plt.subplots(2,1,figsize=(6,7), gridspec_kw={'height_ratios': [3, 1]})
        plt.subplots_adjust(left=0.15,
                            bottom=0.1,
                            right=0.9,
                            top=0.95,
                            wspace=0,
                            hspace=0.3)
    else:
        plt.cla()
    int_ax.set_xlim(-5, 5)
    int_ax.set_ylim(-5, 5)
    int_ax.grid(True)  # Display a grid

    # Initialize the robot and heading arrow
    plt_robot = Circle((initial[0][0], initial[0][1]), 0.5*scale, fill=True)
    int_ax.add_patch(plt_robot)

    # Initialize the trajectory line
    plt_line, = int_ax.plot([], [], 'b-')

    # Mark the goal position as a cross ('x') and add a heading arrow
    goal, = int_ax.plot(goalpos[0][0], goalpos[0][1], 'rx')
    goal_heading = FancyArrow(goalpos[0][0], goalpos[0][1],
                                   np.cos(goalpos[1][2])*scale,
                                   np.sin(goalpos[1][2])*scale, color='red', width=0.3*scale)
    int_ax.add_patch(goal_heading)

    rpe_arrow = FancyArrow(initial[0][0], initial[0][1],
                           goalpos[0][0]-initial[0][0], goalpos[0][1]-initial[0][1], color='green', width=0.2*scale)
    int_ax.add_patch(rpe_arrow)


    # Initialize the initial pose of the robot and its heading as a faded shape
    plt_initial_pose = Circle((initial[0][0], initial[0][1]), 0.5*scale, fill=True, alpha=0.3)
    plt_initial_heading = FancyArrow(initial[0][0], initial[0][1],
                                          np.cos(initial[1][2])*scale,
                                          np.sin(initial[1][2])*scale, color='gray', width=0.3*scale,
                                          alpha=0.3)


    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    textstr = f"Force: {'{:10.3f}'.format(0)}\n Torque: {'{:10.3f}'.format(0)}"
    # int_text = int_ax.text(0.05, 0.95, textstr, transform=int_ax.transAxes, fontsize=10,
    #             verticalalignment='top', bbox=props)

    int_ax.add_patch(plt_initial_pose)
    int_ax.add_patch(plt_initial_heading)


    legend_len = .1
    legend_elements = [FancyArrow(0, 0, legend_len, 0, color='blue' , label='Robot'),
                       FancyArrow(0, 0, legend_len, legend_len, color='red', label='Goal'),
                       FancyArrow(0, 0, legend_len, 0, color='green', label='$R(p_e)$')]

    int_ax.legend(handles=legend_elements, loc="upper left")
    bar_ax.title.set_text("Controller Terms")
    bar_ax.set_ylabel("Energy")
    rects = bar_ax.bar([r"$\frac{k_p}{2}p_e^{\top}p_e$", r"$\mathcal{V}_{R_1}*\mathcal{V}_{R_2}$",
                        r"$\mathcal{V}_{R_3}$", "Total"], np.ones((4,))*0.5, width=.4,
                       color=['green', 'blue', 'red', 'cyan'])

    # plt.title(str(params))
    return writer, int_fig


def update_interactive_plot(state, force, torque, goalpos, energies):
    global int_fig, int_ax, plt_robot, int_trajx, int_trajy
    pos0, pos1, yaw, vel0, vel1, angvel = state

    int_trajx.append(pos0)
    int_trajy.append(pos1)
    plt_robot.center = pos0, pos1

    # remove all arrows
    for arrow in int_ax.patches:
        if isinstance(arrow, FancyArrow) and arrow is not plt_initial_heading:
            arrow.remove()


    goal_heading = FancyArrow(goalpos[0][0], goalpos[0][1],
                              np.cos(goalpos[1][2])*scale,
                              np.sin(goalpos[1][2]*scale), color='red', width=0.3*scale)
    int_ax.add_patch(goal_heading)

    # new arrow indicating the heading
    heading = FancyArrow(pos0, pos1,
                         np.cos(yaw * np.pi / 180)*scale,
                         np.sin(yaw * np.pi / 180)*scale, color='blue', width=0.3*scale)
    int_ax.add_patch(heading)

    rpe_arrow = FancyArrow(pos0, pos1, goalpos[0][0]-pos0, goalpos[0][1]-pos1, color='green', width=0.2*scale)
    int_ax.add_patch(rpe_arrow)

    # bar_ax.bar(["$.5*k_p||Pe||$", "$V_{R_1}*V_{R_2}$", "$V_{R_3}$", "Total"], energies, width=.4)
    for rect, val in zip(rects,energies):
            rect.set_height(val)
    # textstr = f"Force: {'{:10.3f}'.format(force)}\n Torque: {'{:10.3f}'.format(torque)}"
    # # place a text box in upper left in axes coords
    # int_text.set_text(textstr)
    # update the trajectory line
    plt_line.set_data(int_trajx, int_trajy)
    int_fig.canvas.draw()
    int_fig.canvas.flush_events()
    writer.grab_frame()
    # time.sleep(.01)
