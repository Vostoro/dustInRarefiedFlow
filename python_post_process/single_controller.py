from pathlib import Path
from types import SimpleNamespace

import mesh_reader
import physics

import numpy as np
from scipy.integrate import solve_ivp

def manager(
            mesh_name: str, mesh_path:str = 'python_post_process/VTK/', 
            t_start : int = 0, t_end : int = 1, time_steps : int = 100000, 
            initial_state : np.ndarray = np.array([-0.000213333741212, 4.1, -2, 0.0, 0.0, 0.0], dtype = float)
            ) -> SimpleNamespace:
    '''
        Master function, runs the simulation for the specified mesh and initial conditions, and returns the solution as a SimpleNamespace object.

        mesh_name : str
            Name of the mesh file (with extension) to be used for the simulation.
        mesh_path : str
            Path to the directory containing the mesh file. Default is 'python_post_process/VTK
        t_start : int
            Start time for the simulation in seconds. Default is 0.
        t_end : int
            End time for the simulation in seconds. Default is 1.
        time_steps : int
            Number of time steps for the simulation. Default is 100000.
        initial_state : np.ndarray
            Initial state of the particle, given as a 1D array with 6 elements:
            [x, y, z, vx, vy, vz] in meters and meters per second.

        Returns:
        solution : SimpleNamespace
            An object, that has two pairs: time under key t, and state ([position, velocity]) under key y
        '''

    # Create a MeshReader instance
    print("Reading mesh and preparing interpolators")
    mesh_instance = mesh_reader.MeshReader(path=mesh_path, name=mesh_name)
    leaving_mesh_event = mesh_instance.leaving_mesh_event

    # Get interpolators for velocity, density, and temperature fields
    U, rho, T = mesh_instance.get_interpolators()

    wall = mesh_instance.mesh.extract_surface(algorithm='dataset_surface')    # Not sure what algorithm keyword does, but the default will change in future in pyVIsta, so

    # Compute cell normals for reflection
    wall = wall.compute_normals(
        cell_normals=True,
        point_normals=False
    )

    #### Run the physics simulation
    
    model = physics.PhysicalModel(
        dust_diameter=1e-9,
        dust_density=19300.0,
        dust_temperature=300.0,
        interp_U=U,
        interp_rho=rho,
        interp_T=T,
        signed_distance=mesh_instance.signed_distance
    )


    print("Starting simulation")

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------
    initial_state = initial_state

    t_start = t_start
    t_end = t_end

    time_steps = time_steps
    t_eval = np.linspace(t_start, t_end, time_steps)

    max_step = (t_end - t_start) / (time_steps - 1)

    # Reflection settings
    coefficient_of_restitution = 0.1   # 1.0 = perfectly elastic
    wall_eps = 1e-9                    # small distance used to push back inside
    max_reflections = 10000            # safety limit
    time_tol = 1e-12

    #check: initial point should be inside the domain.
    sd0 = mesh_instance.signed_distance(initial_state[0:3])

    if sd0 < -100.0 * wall_eps:     # Just an arbitrary number to avoid computer zero errors - far enough outside the wall
        raise ValueError(
            f"Initial position is outside the mesh. "
            f"signed_distance = {sd0}"
        )

    # Current state and time
    y = initial_state.copy()    # Since '=' in python would create a shared reference
    current_t = t_start

    # Storage for separate solution pieces
    all_t = []
    all_y = []

    reflection_count = 0

    # main loop
    while current_t < t_end - time_tol:
        sol = solve_ivp(
            model.particle_derives,
            t_span=(current_t, t_end),  # interval of integration
            y0=y,                       # initial state
            method="RK45",
            max_step=max_step,          # limit max_step so that solver does not skip calculataion - beccause he does, bastard (but calculation is slower now)
            events=leaving_mesh_event,  # supply a event_cross event (checks the distance-to-closest-wall sign change)
            dense_output=True,          # !!! compute a continuous solution - keep intermediate calculations inside the time steps - extra RAM, slower, more details  -- HEAVILY AFFECTS THE PLOT, CHECK WHY?
        )

        # If solver failed, stop and keep what already got
        if sol.status == -1:    # status = -1 - integration step failed
            print(f"solve_ivp failed at t ≈ {current_t}: {sol.message}")
            break

        # check whether an out of bounds event
        event_happened = (sol.status == 1)     # status = 1 - a termination event occured

        y_stop = np.empty(1)

        if event_happened:
            t_stop = float(sol.t_events[0][0])          # sol.t_events - for each event type a list of arrays, at which an event happened; [0] - for the first event; [0] - the first time it occured
            y_stop = sol.y_events[0][0]    # for each value of t_events - corresp. value of solution
        else:               ## !!!! CHECK
            # No event - integrate to the end of this segment.
            if sol.t.size == 0:
                t_stop = current_t      # If solver didn't save even one time step - keep the time
            else:
                t_stop = float(sol.t[-1])   # If solver managed a few steps - move to the last step


        
        # save solution values before the event
        mask = (t_eval >= current_t) & (t_eval < t_stop - time_tol) # Getting only new times, since solver can go backwards

        sample_t = t_eval[mask].copy()

        # Saving the solution to the concated one
        try:
            sample_y = sol.sol(sample_t)  
        except ValueError as e:
            print(f"Error evaluating solution at t_eval: {e}")
            print(f"Solution: {sol}")
            break
        all_t.append(sample_t)
        all_y.append(sample_y)

        if not event_happened:
            break

        reflection_count += 1
        if reflection_count > max_reflections:
            print("Maximum number of reflections reached.")
            break

        pos_event = y_stop[0:3]
        vel_event = y_stop[3:6]

        # Reflect
        pos_new, vel_new = model.reflect_at_wall(
            pos_event,
            vel_event,
            wall,
            e=coefficient_of_restitution,
            eps=wall_eps,
        )

        # if the point is still outside, retry, but moving further inside
        local_eps = wall_eps

        for _ in range(8):
            if mesh_instance.signed_distance(pos_new) > 0.0:
                break
            local_eps *= 10.0
            pos_new, vel_new = model.reflect_at_wall(
                pos_event,
                vel_event,
                wall,
                e=coefficient_of_restitution,
                eps=local_eps,
            )   

        # For the next start of the solver - add the reflected position, and start from time step when the calculations stopped
        y = np.empty(6)
        y[0:3] = pos_new
        y[3:6] = vel_new

        current_t = t_stop


    
    # Combine all solution segments, since all_t is a list of solution parts, not a singular array
    t_combined = np.concatenate(all_t)
    y_combined = np.concatenate(all_y, axis=1)

    # Just using namespace instead of disctionary, for fun
    solution = SimpleNamespace(
        t=t_combined,
        y=y_combined,
    )

    print("Simulation complete")

    #if log: logger(sol=solution)
    
    return solution