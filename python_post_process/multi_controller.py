import glob
import single_controller
from single_controller import np

def manager(mesh_template: str, mesh_path: str):
    """
    Manages the processing of multiple meshes based on a naming template.

    Args:
        mesh_template (str): The naming template for the mesh files.
        mesh_path (str): The path where the mesh files are located.

    Returns:
        list: A list of results from processing each mesh.
    """
    # Create a list to store results
    results = []
    times = np.zeros(0)
    coords = np.zeros((0, 3))

    # Use glob to find all files matching the naming template
    name_template = mesh_template
    mesh_files = glob.glob(f"{mesh_path}/{name_template}")

    # Process each mesh file         mesh_name = 'Mine_5000.vtk'
    for mesh_file in sorted(mesh_files, key=lambda file: file.split('_')[-1].replace('.vtk', '')):
        print(f'File: {mesh_file}')
        '''
        # Extract the base name of the mesh file
        base_name = mesh_file.split('/')[-1]
        
        # Call the single_controller manager for each mesh
        mesh_time = pull_mesh_time(mesh_file)  #
        result = single_controller.manager(mesh_name=base_name, mesh_path=mesh_path)
        
        # Append the result to the results list
        results.append(result)

    for result in results:
        times = np.concatenate((times, result.t))
        coords = np.concatenate((coords, result.y[0:3, :].T), axis=0)
    '''
    # return results

def pull_mesh_times(mesh_file: str):
    """
    Extracts the time from the mesh file name.

    Args:
        mesh_file (str): The full path to the mesh file.
    Returns:
        float: The extracted time as a float.
    """
    # Extract the base name of the mesh file
    base_name = mesh_file.split('/')[-1]
    
    # Split the base name to extract the time part
    time_str = base_name.split('_')[-1].replace('.vtk', '')
    
    # Convert the extracted time string to a float
    try:
        time_value = float(time_str)
    except ValueError:
        raise ValueError(f"Could not convert extracted time '{time_str}' to float.")
    
    return time_value