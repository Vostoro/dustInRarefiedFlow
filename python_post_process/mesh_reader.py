from dataclasses import dataclass
from scipy.interpolate import NearestNDInterpolator
import numpy as np
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance
import pyvista as pv

@dataclass
class MeshReader:
    '''
    Class for reading mesh data and creating interpolators for velocity, density, and temperature fields, as well as functions for signed distance and leaving mesh events.
    Attributes:
        path (str): Path to the mesh file.
        name (str): Name of the mesh file.
        mesh (pyVista.DataSet): The mesh data read from the file.
        signned_distance (Callable): Function to compute signed distance from the mesh surface.
        leaving_mesh_event (Callable): Function to detect when a particle leaves the mesh.
    Methods:
        get_interpolators(): Returns interpolators for velocity, density, and temperature fields.
        build_signed_distance(surface, inside_point): Returns a function for signed distance from the mesh surface.
        build_leaving_mesh_event(): Returns a function for detecting when a particle leaves the mesh.
    '''
    path : str
    name : str

    @property
    def signed_distance(self):
        return self._signed_distance
    @signed_distance.setter
    def signed_distance(self, mesh):
        self._signed_distance = self.build_signed_distance()

    def __post_init__(self):
        print('Starting post_init')
        self.mesh = pv.read(self.path + self.name)
        self.wall = self.mesh.extract_surface(algorithm='dataset_surface')    # Not sure what algorithm keyword does, but the default will change in future in pyVIsta, so
        self.wall = self.wall.compute_normals(
        cell_normals=True,
        point_normals=False
        )
        self.inside_point = self.mesh.cell_centers().points[0]
        print('Creating signed distance function')
        self.signed_distance = self.build_signed_distance()
        print('Creating leaving mesh event')
        self.leaving_mesh_event = self.build_leaving_mesh_event()

    def get_interpolators(self):
        """
        Returns interpolators for velocity, density and temperature fields.
        U : numpy.ndarray
            velocity field, shape (N, 3)
        rho : numpy.ndarray
            density field, shape (N,)
        T : numpy.ndarray
            temperature field, shape (N,)
        """
        cell_centers = self.mesh.cell_centers().points
        
        U_gas = self.mesh.cell_data.get('U_He')
        rho_gas = self.mesh.cell_data.get('rhoN_He')        # rhoN is allegedely "Number Density", that is - amount of molecules per cubic meter
        T_gas = self.mesh.cell_data.get('Ttra_He')
        
        
        print('Data reading finished')
        ### Creating data interpolators for gas parameters
        
        print('Interpolating')
        U = NearestNDInterpolator(cell_centers, U_gas)
        rho = NearestNDInterpolator(cell_centers, rho_gas)   
        T = NearestNDInterpolator(cell_centers, T_gas)
        return U, rho, T
    
    def build_signed_distance(self):
        """
        Returns a function signed_distance(p), where:
        > 0 means inside the domain
        = 0 means on the boundary
        < 0 means outside the domain
        surface : pyVista.DataSet
            surface for which to prepare the function
        inside_point : numpy.ndarray
            a point definitely inside the surface, to calibrate the direction of distance

        """
        inside_point = np.asarray(self.inside_point, dtype=float)

        implicit = vtkImplicitPolyDataDistance()    # Function that computes distance from a point x to nearest point p on an input vtkPolyData
        implicit.SetInput(self.wall)          # sets input vtkPolyData

        d0 = implicit.EvaluateFunction(     # Evaluates the function (distance) at position x-y-z
            float(inside_point[0]),
            float(inside_point[1]),
            float(inside_point[2])
        )

        if abs(d0) < 1e-12:
            raise ValueError(
                "The calibration point appears to lie on the surface. "
                "Choose a point clearly inside the domain."
            )

        # Make the known inside point have positive signed distance, to asses how does it handle signs of distance
        sign_correction = np.sign(d0)

        def signed_distance(p):
            p = np.asarray(p, dtype=float)
            d = implicit.EvaluateFunction(
                float(p[0]),
                float(p[1]),
                float(p[2])
            )
            return float(sign_correction * d)

        return signed_distance

    def build_leaving_mesh_event(self):
        """
        Returns a function leaving_mesh_event(t, y), where:
        > 0 means inside the domain
        = 0 means on the boundary
        < 0 means outside the domain
        t : float
            time; required for the solver
        y : numpy.ndarray
            numpy array, containing position to check distance for
        """
        def leaving_mesh_event(t, y):
            pos = y[0:3]
            return self.signed_distance(pos)

        leaving_mesh_event.terminal = True  # type: ignore # Sets parameter for solver: whether to terminate when event triggers, implicitly False. If int - terminates after specified amount of events occured
        leaving_mesh_event.direction = -1   # type: ignore # Setc parameter for solver: direction of 0 crossing. If positive - triggers the event when going from negative to positive; if negative - vice versa, if 0 - both, implicitly 0 if not assigned

        return leaving_mesh_event
