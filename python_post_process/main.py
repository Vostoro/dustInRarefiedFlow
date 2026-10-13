import single_controller, plotter, mesh_reader





if __name__ == "__main__":
    
    name_template = 'Mine_*.vtk'
    mesh_path = 'python_post_process/VTK/'
    mesh_name = 'Mine_5000.vtk'

    mesh = mesh_reader.MeshReader(path=mesh_path, name=mesh_name)
    mesh = mesh.mesh
    solve = single_controller.manager(
        mesh_name=mesh_name,
        mesh_path=mesh_path,
    )

    plotter.visualiser(solve, mesh)
