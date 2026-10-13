from matplotlib import pyplot as plt
import numpy as np
import pyvista as pv

def visualiser(solution, mesh):
    '''
    Plotting function, plots particle trajectory projections at 3 porjections: xy, xz, yz, as well as time-dependance of each coordinates 
    sol : SimpleNamespace
        An object, that has two pairs: time under key t, and state ([position, velocity]) under key y
    mesh : PolyData
        Dataset consisting of the mesh of the 3-d model in which to plot the trajectory 
    '''
    print('Plotting start')
    x_traj = solution.y[0]
    y_traj = solution.y[1]
    z_traj = solution.y[2]
    times = solution.t

    fig, axs = plt.subplots(1, 3)
    fig.suptitle("Tungsten dust trajectory in Rarefied Gas")
    axs[0].plot(x_traj * 1e6, y_traj * 1e6, label="Particle Trajectory", color='red')
    axs[0].set_title('x, y coordinates of dust')
    axs[0].set(xlabel='x coord, micron', ylabel='y coord, micron')
    axs[0].grid(True)

    axs[1].plot(x_traj * 1e6, z_traj * 1e6, label="Particle Trajectory", color='blue')
    axs[1].set_title('x, z coordinates of dust')
    axs[1].set(xlabel='x coord, micron', ylabel='z coord, micron')
    axs[1].grid(True)

    axs[2].plot(z_traj * 1e6, y_traj * 1e6,  label="Particle Trajectory", color='blue')
    axs[2].set_title('z, y coordinates of dust')
    axs[2].set(xlabel='z coord, micron', ylabel='y coord, micron')
    axs[2].grid(True)

    plt.show()

    fig, axs = plt.subplots(1, 3)
    fig.suptitle("Tungsten dust trajectory over time in Rarefied Gas")
    axs[0].plot(times, x_traj * 1e6, color='red')
    axs[0].set_title('x coordinate of dust')
    axs[0].set(ylabel='x coord, micron', xlabel='Time, sec')
    axs[0].grid(True)

    axs[1].plot(times, y_traj * 1e6, color='blue')
    axs[1].set_title('y coordinate of dust')
    axs[1].set(ylabel='y coord, micron', xlabel='Time, sec')
    axs[1].grid(True)

    axs[2].plot(times, z_traj * 1e6, color='green')
    axs[2].set_title('z coordinate of dust')
    axs[2].set(ylabel='z coord, micron', xlabel='Time, sec')
    axs[2].grid(True)

    plt.show()
    print('Plotting finished')

    # --- 3D PyVista Plotting ---
    print('Plotting 3D start')
    points = np.column_stack((x_traj, y_traj, z_traj))
    
    # Create line cells connecting trajectory points
    trajectory_line = pv.lines_from_points(points)

    plotter = pv.Plotter()

    # Add VTK Mesh Surface with Transparency/Opacity
    plotter.add_mesh(mesh, color='lightblue', opacity=0.3, show_edges=True, label='Geometry Mesh')

    # Add Trajectory Line
    plotter.add_mesh(trajectory_line, color='red', line_width=4, label='Particle Trajectory')

    # Optional: Highlight Start and End Points
    plotter.add_mesh(pv.PolyData([points[0]]) , color='green', point_size=12, render_points_as_spheres=True, label='Start')
    plotter.add_mesh(pv.PolyData([points[-1]]), color='yellow', point_size=12, render_points_as_spheres=True, label='End')

    plotter.add_legend()    # type: ignore
    plotter.add_axes()      # type: ignore
    plotter.show_grid()     # type: ignore
    
    # Display interactive 3D window
    plotter.show()
    print('Plotting finished')