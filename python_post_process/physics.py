from dataclasses import dataclass
from collections.abc import Callable
from typing import Tuple

import numpy as np
from scipy.special import erf
from scipy.interpolate import NearestNDInterpolator




@dataclass
class PhysicalModel:
    dust_diameter: float
    dust_density : float
    dust_temperature: float
    interp_U: NearestNDInterpolator
    interp_rho: NearestNDInterpolator
    interp_T: NearestNDInterpolator
    signed_distance: Callable

    def __post_init__(self):
        self.molecular_mass, self.collision_diameter, self.heat_capacity_ratio, self.gas_constant = self.set_constants('He')
        self.dust_surface = np.pi * (self.dust_diameter / 2) ** 2
        dust_volume = (np.pi/6) * self.dust_diameter**3
        self.dust_mass = dust_volume * self.dust_density
    
    def set_constants(self, gas : 'str') -> Tuple[float,float,float,float]:
        '''
        Function that returns 4 gas parameters : Mass of one molecule, collisional diameter, heat capacity ratio, specific gas constant
        '''
        Na = 6.022e23                        # Avogadro's number
        if gas == 'N':
            Molar_M = 0.028013                  # Molar mass of N2, in kg/mol
            M = Molar_M / Na                    # Mass of one molecule of N2, in kg
            D_col = 3.64e-10                    # N2 collisional diameter
            Gamma = 1.47                         # Heat capasity ratio for N2
            R_gas = 8.31/Molar_M                # Specific gas  constant for N2 (J/kg*K)
        elif gas == 'He':   # He
            Molar_M = 0.004                     # Molar mass of He, in kg/mol
            M = Molar_M / Na
            D_col = 2.6e-10                    # He collisional diameter
            Gamma = 1.66                         # Heat capasity ratio for He
            R_gas = 8.31/Molar_M                       # Specific gas  constant for He (J/kg*K)
        elif gas == 'air': # air
            Molar_M = 0.02897                   # Molar mass of air, in kg/mol
            M = Molar_M / Na
            D_col = 3.7e-10                     # Air collisional diameter
            Gamma = 1.4                          # Heat capasity ratio for air
            R_gas = 8.31/Molar_M                 # Specific gas  constant for air (J/kg*K)
        else: # water
            Molar_M = 0.018                     # Molar mass of water, in kg/mol
            M = Molar_M / Na
            D_col = 2.75e-10                    # Water collisional diameter
            Gamma = 1.33                         # Heat capasity ratio for water
            R_gas = 8.31/Molar_M                 # Specific gas  constant for water (J/kg*K)

        return M, D_col, Gamma, R_gas

    def particle_derives(self, t : float, y : np.ndarray):
        """Return the particle's velocity and acceleration at state ``y``."""
        pos = y[0:3]
        vel = y[3:6]

        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel)):
            return np.zeros(6)

        if self.signed_distance(pos) < -1e-10:
            return np.concatenate([vel, np.zeros(3)])

        u_g = self.interp_U(pos)[0]
        T_g = self.interp_T(pos)[0]
        n_g = self.interp_rho(pos)[0]

        if np.isnan(T_g) or np.any(np.isnan(u_g)) or np.isnan(n_g):
            return np.concatenate([vel, np.zeros(3)])

        v_rel_vec = u_g - vel
        F_drag, _ = self.calculate_drag_force(v_rel_vec, T_g, n_g)
        F_gravity = np.array([0.0, 0.0, -9.81 * self.dust_mass])

        acc = (F_drag + F_gravity) / self.dust_mass

        return [vel[0], vel[1], vel[2], acc[0], acc[1], acc[2]]

    def reflect_at_wall(self, pos, vel, surface, e=1.0, eps=1e-9):
        """
        Returns position and velocity; position moved deeper inside, normal velocity component reflected
        pos : array_like
            current position,  [x,y,z]
        vel : array_like
            current_velocity, [Vx,Vy,Vz]
        surface : 
            pyvista DataSet, containing points of the wall, that has been intersected
        e : 
            float, coefficient of restitution

        Returns
        pos : ndarray
            position back inside
        vel : ndarray
            velocity with reflected normal component
        """
        pos = np.asarray(pos)
        vel = np.asarray(vel)

        # find closest wall cell
        cell_id = surface.find_closest_cell(pos)

        # Get normal of that wall cell
        n = self.get_cell_normal(surface, cell_id)

        # Orient normal outward
        # Since signed_distance > 0 inside, moving outward along n should give signed_distance < 0.
        if self.signed_distance(pos + eps * n) > 0: # Since signed_distance >0 when the point is inside - reoriend the normal
            n = -n

        # Normal component of velocity relative to outward normal
        vn = np.dot(vel, n) # X * Y * cos(theta)

        # Reflect only if moving outward
        if vn > 0:
            vel = vel - (1.0 + e) * vn * n  # Multiplying by n, to convert scalar vn to 3-d vector; (1 + e) - 1 is to cancel the current speed; e  - speed inwards

        #if np.linalg.norm(vel) < calculate_adhesion_velocity(): # If the velocity is below adhesion threshold - stop the particle
            #vel = np.zeros(3)
            #print(f"Adhesion velocity: {calculate_adhesion_velocity()}")
        #print(f"Yield velocity: {calculate_yield_velocity()}")
            #print(f"Particle has velocity: {np.linalg.norm(vel)}")
        # Push the particle slightly back inside the domain
        pos = pos - 2.0 * eps * n

        return pos, vel

    def get_cell_normal(self, surface, cell_id):      
        """
        Returns normalised normal to the [surface] vectorat the cell with [cell_id]
        surface : pyVista.DataSet
            surface for which to build normal
        cell_id : int
            id of the cell from the surface to build normal to.

        Returns
        val : ndarray
            normalised vectot, perpendicular to the surface; direction (inward or outward) is not guaranteed
        """
        n = np.asarray(surface.cell_data["Normals"][cell_id], dtype=float)
        norm = np.linalg.norm(n)
        if norm > 1e-15:
            return n / norm
        else:
            raise ValueError(f"Normal vector at cell {cell_id} has zero length.")

    def calculate_drag_force(self, v_rel_vec, T_g, n_g, log_o=False):
        """Calculate the drag force vector and drag coefficient on the particle."""
        v_rel = np.linalg.norm(v_rel_vec)

        rho_g = n_g * self.molecular_mass
        mu_g = 1.78e-5 * (T_g / 273.15) ** 0.75

        a_g = np.sqrt(self.heat_capacity_ratio * self.gas_constant * T_g)
        Ma = v_rel / a_g
        Re = (rho_g * v_rel * self.dust_diameter) / mu_g
        Re = max(Re, 1e-10)
        Re = min(1e4, Re)

        lambda_g = 1.0 / (
            np.sqrt(2) * np.pi * self.collision_diameter**2 * n_g
        )
        Kn = lambda_g / self.dust_diameter
        Kn = min(Kn, 1e8)

        if Kn > 10:
            C_D = self.calculate_free_mol_coefficient(Ma, T_g, self.heat_capacity_ratio)
        elif Kn < 0.1:
            C_D = self.calculate_continuum_coefficient(Re, Ma, Kn)
        else:
            C_D = self.calculate_drag_coefficient_trans(
                Re, Ma, T_g, self.dust_temperature
            )
        #with warnings.catch_warnings(record=True) as w:
         #   warnings.simplefilter(action = 'default', category=RuntimeWarning)
          #  if not not w:
           #     print(f'w = {w}')
            #    if 'overflow' in str(w[-1].message):
             #       print(f"Overflow warning in drag coefficient calculation: rho_g={rho_g}, v_rel={v_rel}, C_D={C_D}, dust_surface={self.dust_surface}")
        F_drag_mag = 0.5 * rho_g * v_rel**2 * C_D * self.dust_surface
        F_drag_vec = F_drag_mag * (v_rel_vec / v_rel)

        return F_drag_vec, C_D

    def calculate_free_mol_coefficient(self, Ma, T_g, Gamma):
        S = Ma * np.sqrt(Gamma / 2.0)
        S_w = S * np.sqrt(T_g / self.dust_temperature)
        if S < 0.01:
            val = 35 / (6 * np.sqrt(np.pi) * S) + 2 * np.sqrt(np.pi) / (3 * S_w)
        else:
            val = (
                (1 + 2 * S**2) * np.exp(-(S**2) / 2) / (np.sqrt(np.pi) * S**3)
                + erf(S) * (4 * S**4 + 4 * S**2 - 1) / (2 * S**4)
                + 2 * np.sqrt(np.pi) / (3 * S_w)
            )

        return val

    def calculate_continuum_coefficient(self, Re, Ma, Kn):
        if Ma < 1:
            if Re < 1.0:
                val = 24.0 / Re + 4.5
            else:
                val = (24.0 / Re) * (1.0 + 0.15 * Re**0.687)
        else: # Ma >= 1:
            val = (0.97 + 1.32 / np.sqrt(Re)) * ((1.0 + 1.0 / np.sqrt(Re)) ** 2)
        # if Kn > 0.001:
        A1, A2, A3 = 1.257, 0.4, 1.1
        Cc = 1.0 + Kn * (A1 + A2 * np.exp(-A3 / Kn))
        val = val / Cc
        return val

    def calculate_drag_coefficient_trans(self, Re, Ma, T_g, T_w, log=False):
        S = Ma * np.sqrt(1.47 / 2.0)
        if Ma < 1:
            C_D = self.low_ma_trans(Re, Ma, S, T_g, T_w)
        elif Ma >= 1.75:
            C_D = self.high_ma_trans(Re, Ma, S, T_g, T_w)
        else:
            C_D = self.low_ma_trans(Re, 1.0, S, T_g, T_w) + 4 / 3 * (Ma - 1) * (
                self.high_ma_trans(Re, 1.75, S, T_g, T_w)
                - self.low_ma_trans(Re, 1.0, S, T_g, T_w)
            )

        return C_D

    def low_ma_trans(self, Re, Ma, S, T_g, T_w):
        # Henderson C. B. Drag Coefficients of Spheres in Continuum and Rarefied Flows. VOL. 14, NO. 6, JUNE 1976 AIAA JOURNAL
        return (
            24 * ((Re + S * (4.33 + (  (3.65-1.53*T_w/T_g) / (1 + 0.353*T_w/T_g)  ) * np.exp(-1 * 0.247 * Re / S))) ** (-1)) + \
            np.exp( -1 * 0.5 * Ma / np.sqrt(Re)) * (  (4.5 + 0.38 * (0.03 * Re + 0.48 * np.sqrt(Re))) / (1 + 0.03 * Re + 0.48 * np.sqrt (Re)) + 0.1 * Ma ** 2 + 0.2 * (Ma ** 8)  ) + \
                (1 - np.exp (-1 * Ma / Re)) * 0.6 * S
        )

    def high_ma_trans(self, Re, Ma, S, T_g, T_w):
        # Henderson C. B. Drag Coefficients of Spheres in Continuum and Rarefied Flows. VOL. 14, NO. 6, JUNE 1976 AIAA JOURNAL
        return (0.9 + 0.34 / (Ma**2) + 1.86 * np.sqrt(Ma / Re) * (2 + 2 / S + 1.058 / S * np.sqrt(T_w/T_g) - 1/ (S ** 4))) / \
                (1 + 1.86 * np.sqrt(Ma / Re)
        ) #val = (2 + 2 / S + 1.058 / S * np.sqrt(T_w/T_g) - 1/ (S ** 4))  # Simplified equation for molecular flow
    