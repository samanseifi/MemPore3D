from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
from scipy import fft as spfft

import numba


class PETScPoissonGAMG:
    def __init__(self, Nx, Ny, Nz, dx, dy, dz):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz if Nz % 2 else Nz + 1
        self.dx2, self.dy2, self.dz2 = dx*dx, dy*dy, dz*dz
        self.cx, self.cy, self.cz = 1.0/self.dx2, 1.0/self.dy2, 1.0/self.dz2

        self.k_mem_minus = self.Nz // 2 - 1
        self.k_mem_plus  = self.Nz // 2 + 1

        N = self.Nx * self.Ny * self.Nz
        self.A = PETSc.Mat().create(comm=self.comm)
        self.A.setSizes(((PETSc.DECIDE, N), (PETSc.DECIDE, N)))
        self.A.setType(PETSc.Mat.Type.MPIAIJ)
        self.A.setPreallocationNNZ((7,7))
        self.A.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        self.A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

        self.b = PETSc.Vec().createMPI(N, comm=self.comm); self.b.set(0.0)
        self.x = PETSc.Vec().createMPI(N, comm=self.comm); self.x.set(0.0)

        r0, r1 = self.A.getOwnershipRange()
        for row in range(r0, r1):
            i = row % self.Nx
            t = row // self.Nx
            j = t % self.Ny
            k = t // self.Ny

            diag = 0.0; cols=[]; vals=[]
            # X-neighbors (Neumann)
            if i == 0:
                cols.append(self.gid(i+1,j,k)); vals.append(-2.0*self.cx); diag += 2.0*self.cx
            elif i == self.Nx-1:
                cols.append(self.gid(i-1,j,k)); vals.append(-2.0*self.cx); diag += 2.0*self.cx
            else:
                cols += [self.gid(i-1,j,k), self.gid(i+1,j,k)]
                vals += [-self.cx, -self.cx]; diag += 2.0*self.cx

            # Y-neighbors (Neumann)
            if j == 0:
                cols.append(self.gid(i,j+1,k)); vals.append(-2.0*self.cy); diag += 2.0*self.cy
            elif j == self.Ny-1:
                cols.append(self.gid(i,j-1,k)); vals.append(-2.0*self.cy); diag += 2.0*self.cy
            else:
                cols += [self.gid(i,j-1,k), self.gid(i,j+1,k)]
                vals += [-self.cy, -self.cy]; diag += 2.0*self.cy

            # Z-neighbors (interior stencil for now, Dirichlet BCs handled by zeroRowsColumns)
            # This robustly adds the full diagonal contribution before BCs are applied.
            if k > 0:
                cols.append(self.gid(i,j,k-1)); vals.append(-self.cz)
            if k < self.Nz-1:
                cols.append(self.gid(i,j,k+1)); vals.append(-self.cz)
            diag += 2.0*self.cz # Full diagonal term for Z

            cols.append(row); vals.append(diag)
            self.A.setValues([row], cols, vals)

        self.A.assemblyBegin(); self.A.assemblyEnd()
        self.b.assemblyBegin(); self.b.assemblyEnd()
        
        self.A_pristine = self.A.copy()

        self.ksp = PETSc.KSP().create(comm=self.comm)
        self.ksp.setOperators(self.A)
        self.ksp.setType('cg')
        pc = self.ksp.getPC(); pc.setType('gamg')
        self.ksp.setTolerances(rtol=1e-6, max_it=200)
        self.ksp.setFromOptions()

        self.rows_z0   = [self.gid(i,j,0) for j in range(self.Ny) for i in range(self.Nx)]
        self.rows_z1   = [self.gid(i,j,self.Nz-1) for j in range(self.Ny) for i in range(self.Nx)]
        self.rows_km   = [self.gid(i,j,self.k_mem_minus) for j in range(self.Ny) for i in range(self.Nx)]
        self.rows_kp   = [self.gid(i,j,self.k_mem_plus) for j in range(self.Ny) for i in range(self.Nx)]
        self.IS_z0 = PETSc.IS().createGeneral(self.rows_z0, comm=self.comm)
        self.IS_z1 = PETSc.IS().createGeneral(self.rows_z1, comm=self.comm)
        self.IS_km = PETSc.IS().createGeneral(self.rows_km, comm=self.comm)
        self.IS_kp = PETSc.IS().createGeneral(self.rows_kp, comm=self.comm)

    def gid(self, i, j, k):
        return i + self.Nx * (j + self.Ny * k)

    def apply_dirichlet_planes(self, Vm, V_applied):
        """
        Set boundary values into x and zero rows+cols with unit diag.
        """
        self.A.destroy()
        self.A = self.A_pristine.copy()
        self.ksp.setOperators(self.A)
        self.x.set(0.0); self.b.set(0.0)

        for r in self.rows_z0: self.x.setValue(r, -0.5 * V_applied)
        for r in self.rows_z1: self.x.setValue(r, +0.5 * V_applied)
        
        if Vm is not None:
            for j in range(self.Ny):
                for i in range(self.Nx):
                    r_minus = self.rows_km[i + self.Nx*j]
                    r_plus  = self.rows_kp[i + self.Nx*j]
                    self.x.setValue(r_minus, -0.5 * Vm[i, j])
                    self.x.setValue(r_plus,  +0.5 * Vm[i, j])

        self.x.assemble(); self.b.assemble()

        self.A.zeroRowsColumns(self.IS_z0, 1.0, self.x, self.b)
        self.A.zeroRowsColumns(self.IS_z1, 1.0, self.x, self.b)
        self.A.zeroRowsColumns(self.IS_km, 1.0, self.x, self.b)
        self.A.zeroRowsColumns(self.IS_kp, 1.0, self.x, self.b)

    def solve(self):
        self.ksp.solve(self.b, self.x)
        return self.x

    def current_slice_kplus_minus(self, dz, sigma_e):
        """
        Compute J_z at k_mem_plus and k_mem_minus. Gathers full vector to rank 0.
        Returns a tuple of (J_plus, J_minus) as (Nx, Ny) ndarrays on rank 0; returns None elsewhere.
        """
        scat, y = PETSc.Scatter.toZero(self.x)
        scat.begin(self.x, y, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        scat.end(self.x, y, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        scat.destroy()

        if self.rank == 0:
            phi_flat = y.getArray(readonly=True)
            phi = phi_flat.reshape(self.Nz, self.Ny, self.Nx).transpose(2, 1, 0)
            k_plus = self.k_mem_plus
            k_minus = self.k_mem_minus
            grad_plus = (phi[:, :, k_plus + 1] - phi[:, :, k_plus]) / dz
            grad_minus = (phi[:, :, k_minus] - phi[:, :, k_minus - 1]) / dz
            return sigma_e * (grad_plus + grad_minus)
        else:
            return None
        
        
# class SpectralPoissonSolver_old:
#     """
#     Fast spectral 3D Poisson/Laplace solver using:
#     - DCT-II in X and Y directions (for Neumann BCs)
#     - Analytical solution in Z direction (for Dirichlet BCs)

#     This solver is ~10-50x faster than iterative GAMG for this geometry.
#     It provides a drop-in replacement for PETScPoissonGAMG.

#     Boundary Conditions:
#     - X, Y: Homogeneous Neumann (∂φ/∂n = 0)
#     - Z: Non-homogeneous Dirichlet at 4 planes:
#         * z=0: φ = -0.5*V_applied
#         * z=k_mem_minus: φ = -0.5*Vm[i,j]
#         * z=k_mem_plus: φ = +0.5*Vm[i,j]
#         * z=Nz-1: φ = +0.5*V_applied

#     The domain is split into 3 Z-regions, and each is solved analytically
#     in spectral space using sinh/cosh basis functions.
#     """
#     def __init__(self, Nx, Ny, Nz, dx, dy, dz):
#         self.Nx, self.Ny = Nx, Ny
#         self.Nz = Nz if Nz % 2 else Nz + 1
#         self.dx, self.dy, self.dz = dx, dy, dz

#         # Membrane plane indices
#         self.k_mem_minus = self.Nz // 2 - 1
#         self.k_mem_plus = self.Nz // 2 + 1

#         # Precompute wavenumbers for DCT-II (Neumann BCs)
#         # For Neumann BCs with DCT-II: k_n = π*n/L, n=0,1,2,...
#         kx = np.arange(Nx)
#         ky = np.arange(Ny)
#         Lx, Ly = Nx * dx, Ny * dy

#         lambda_x = np.pi * kx / Lx
#         lambda_y = np.pi * ky / Ly

#         # Create 2D grid of eigenvalues
#         self.lambda2 = lambda_x[:, np.newaxis]**2 + lambda_y[np.newaxis, :]**2

#         # Storage for solution
#         self.phi = np.zeros((Nx, Ny, Nz), dtype=np.float64)

#         # Boundary condition storage
#         self.bc_z0 = None
#         self.bc_z1 = None
#         self.bc_km = None
#         self.bc_kp = None

#         # For compatibility with PETSc interface
#         self.comm = MPI.COMM_WORLD
#         self.rank = self.comm.Get_rank()

#         # Warning: This is a serial solver (runs on rank 0 only)
#         if self.rank == 0:
#             print("INFO: Using SpectralPoissonSolver (serial, runs on rank 0 only)")

#     def apply_dirichlet_planes(self, Vm, V_applied):
#         """Set boundary conditions (same interface as PETScPoissonGAMG)."""
#         if self.rank == 0:
#             self.bc_z0 = -0.5 * V_applied * np.ones((self.Nx, self.Ny))
#             self.bc_z1 = +0.5 * V_applied * np.ones((self.Nx, self.Ny))

#             if Vm is not None:
#                 self.bc_km = -0.5 * Vm.copy()
#                 self.bc_kp = +0.5 * Vm.copy()
#             else:
#                 self.bc_km = np.zeros((self.Nx, self.Ny))
#                 self.bc_kp = np.zeros((self.Nx, self.Ny))

#         # Broadcast to all ranks for consistency
#         self.bc_z0 = self.comm.bcast(self.bc_z0, root=0)
#         self.bc_z1 = self.comm.bcast(self.bc_z1, root=0)
#         self.bc_km = self.comm.bcast(self.bc_km, root=0)
#         self.bc_kp = self.comm.bcast(self.bc_kp, root=0)

#     def solve(self):
#         """
#         Solve the 3D Laplace equation ∇²φ = 0 using spectral methods.

#         Strategy:
#         1. Apply 2D DCT in X-Y plane to boundary conditions
#         2. For each (kx, ky) mode, solve analytically in Z direction
#         3. Apply inverse 2D DCT to get physical space solution

#         The Z-direction is divided into 3 regions:
#         - Region 1: [0, k_mem_minus]
#         - Region 2: [k_mem_minus, k_mem_plus] (membrane, linear interpolation)
#         - Region 3: [k_mem_plus, Nz-1]

#         Returns: self (for compatibility with PETSc interface)
#         """
#         if self.rank == 0:
#             # Transform boundary conditions to spectral space
#             bc_z0_hat = spfft.dctn(self.bc_z0, type=2, norm='ortho')
#             bc_z1_hat = spfft.dctn(self.bc_z1, type=2, norm='ortho')
#             bc_km_hat = spfft.dctn(self.bc_km, type=2, norm='ortho')
#             bc_kp_hat = spfft.dctn(self.bc_kp, type=2, norm='ortho')

#             # Allocate spectral space solution
#             phi_hat = np.zeros((self.Nx, self.Ny, self.Nz), dtype=np.float64)

#             # Set boundary values in spectral space
#             phi_hat[:, :, 0] = bc_z0_hat
#             phi_hat[:, :, self.k_mem_minus] = bc_km_hat
#             phi_hat[:, :, self.k_mem_plus] = bc_kp_hat
#             phi_hat[:, :, -1] = bc_z1_hat

#             # Solve for each (kx, ky) mode
#             for ikx in range(self.Nx):
#                 for iky in range(self.Ny):
#                     lambda_k = np.sqrt(self.lambda2[ikx, iky])

#                     # Region 1: z ∈ [0, k_mem_minus]
#                     _solve_region_analytical(
#                         phi_hat, ikx, iky, lambda_k,
#                         z_start=0, z_end=self.k_mem_minus, dz=self.dz,
#                         phi_left=bc_z0_hat[ikx, iky],
#                         phi_right=bc_km_hat[ikx, iky]
#                     )

#                     # Region 2: z ∈ [k_mem_minus, k_mem_plus] (membrane, just 2-3 points)
#                     # Use linear interpolation for this thin region
#                     n_mem = self.k_mem_plus - self.k_mem_minus
#                     for iz in range(1, n_mem):
#                         alpha = iz / n_mem
#                         phi_hat[ikx, iky, self.k_mem_minus + iz] = \
#                             (1 - alpha) * bc_km_hat[ikx, iky] + alpha * bc_kp_hat[ikx, iky]

#                     # Region 3: z ∈ [k_mem_plus, Nz-1]
#                     _solve_region_analytical(
#                         phi_hat, ikx, iky, lambda_k,
#                         z_start=self.k_mem_plus, z_end=self.Nz - 1, dz=self.dz,
#                         phi_left=bc_kp_hat[ikx, iky],
#                         phi_right=bc_z1_hat[ikx, iky]
#                     )

#             # Inverse 2D DCT for each Z-slice to get back to physical space
#             for iz in range(self.Nz):
#                 self.phi[:, :, iz] = spfft.idctn(phi_hat[:, :, iz], type=2, norm='ortho')

#         # Broadcast solution to all ranks
#         self.phi = self.comm.bcast(self.phi, root=0)

#         return self

#     def current_slice_kplus_minus(self, dz, sigma_e):
#         """
#         Compute ionic current density at membrane planes.
#         Same interface as PETScPoissonGAMG.

#         Returns J = σ_e * (∂φ/∂z)|_{k+} + (∂φ/∂z)|_{k-}
#         """
#         if self.rank == 0:
#             k_plus = self.k_mem_plus
#             k_minus = self.k_mem_minus

#             # Finite difference gradients
#             # grad_plus = (self.phi[:, :, k_plus + 1] - self.phi[:, :, k_plus]) / dz
#             # grad_minus = (self.phi[:, :, k_minus] - self.phi[:, :, k_minus - 1]) / dz
#             dphidz_plus  = (-3*self.phi[:,:,k_plus]  + 4*self.phi[:,:,k_plus+1]  - self.phi[:,:,k_plus+2]) /(2*dz)
#             dphidz_minus = ( 3*self.phi[:,:,k_minus] - 4*self.phi[:,:,k_minus-1] + self.phi[:,:,k_minus-2])/(2*dz)
#             J = sigma_e * (dphidz_plus + dphidz_minus)

            
#             return J
#         else:
#             return None
        
# @numba.njit(cache=True, fastmath=True)
# def _solve_region_analytical(phi_hat, ikx, iky, lambda_k,
#                                 z_start, z_end, dz, phi_left, phi_right):
#     """
#     Solve 1D Helmholtz eq: d^2 phi / dz^2 - lambda^2 phi = 0
#     Using numerically stable formulation for large lambda.
#     """
#     # Create local coordinate system
#     n_points = z_end - z_start + 1
#     z_local = np.arange(n_points) * dz
#     L_region = (n_points - 1) * dz
    
#     # Select the slice of the solution array we are writing to
#     # (This avoids the slow Python loop 'for iz in range...')
#     sol_slice = phi_hat[ikx, iky, z_start : z_end + 1]

#     # Case 1: Zero frequency (Linear interpolation)
#     if lambda_k < 1e-12:
#         alpha = z_local / L_region
#         sol_slice[:] = (1.0 - alpha) * phi_left + alpha * phi_right
#         return

#     # Case 2: High frequency (Numerically Stable Formulation)
#     # Formula: phi(z) = phi_L * sinh(lam*(L-z))/sinh(lam*L) + phi_R * sinh(lam*z)/sinh(lam*L)
    
#     # Check for potential overflow in sinh (limit is ~709 for float64)
#     # For a 10um box, lambda can get high, but usually < 700 with standard grids.
#     # If lambda_k * L_region > 700, we should use exp formulation, 
#     # but standard np.sinh handles up to ~10^300, so checking for 'inf' is usually enough.
    
#     arg_L = lambda_k * L_region
    
#     # If argument is too large, use asymptotic approximation to avoid Inf/NaN
#     if arg_L > 700: 
#         # approximate sinh(x) ~ exp(x)/2
#         # ratio sinh(lam*(L-z)) / sinh(lam*L) ~ exp(-lam*z)
#         term1 = phi_left * np.exp(-lambda_k * z_local)
#         # ratio sinh(lam*z) / sinh(lam*L) ~ exp(-lam*(L-z))
#         term2 = phi_right * np.exp(-lambda_k * (L_region - z_local))
#         sol_slice[:] = term1 + term2
#     else:
#         denom = np.sinh(arg_L)
#         term1 = phi_left * np.sinh(lambda_k * (L_region - z_local)) / denom
#         term2 = phi_right * np.sinh(lambda_k * z_local) / denom
#         sol_slice[:] = term1 + term2
        

        
import numpy as np
import scipy.fft as spfft
from mpi4py import MPI
import numba

class SpectralPoissonSolver:
    """
    3D Electrostatic Solver using a singular-interface jump condition.
    Enforces current continuity (Flux+) = (Flux-) and Potential Jump (Phi+ - Phi- = Vm).
    """
    def __init__(self, Nx, Ny, Nz, dx, dy, dz):
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.Lz = (Nz - 1) * dz

        # Membrane location (Midplane)
        self.k_mem_minus = self.Nz // 2 - 1
        self.k_mem_plus  = self.Nz // 2 + 1

        # Precompute wavenumbers
        kx = np.pi * np.arange(Nx) / (Nx * dx)
        ky = np.pi * np.arange(Ny) / (Ny * dy)
        self.lambda2 = kx[:, np.newaxis]**2 + ky[np.newaxis, :]**2

        self.phi = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

    def apply_dirichlet_planes(self, Vm, V_applied):
        """Prepares spectral boundary conditions."""
        if self.rank == 0:
            # External Electrodes
            self.bc_top = 0.5 * V_applied * np.ones((self.Nx, self.Ny))
            self.bc_bot = -0.5 * V_applied * np.ones((self.Nx, self.Ny))
            # Membrane Jump
            self.Vm = Vm.copy() if Vm is not None else np.zeros((self.Nx, self.Ny))

    def solve(self):
        """Solves Laplace eq with interface jump and flux continuity using stable exp scaling."""
        if self.rank != 0: return self

        top_hat = spfft.dctn(self.bc_top, type=2, norm='ortho')
        bot_hat = spfft.dctn(self.bc_bot, type=2, norm='ortho')
        Vm_hat  = spfft.dctn(self.Vm,     type=2, norm='ortho')

        phi_hat = np.zeros((self.Nx, self.Ny, self.Nz), dtype=np.float64)
        L = self.Lz / 2.0 

        for ikx in range(self.Nx):
            for iky in range(self.Ny):
                lam = np.sqrt(self.lambda2[ikx, iky])
                
                if lam < 1e-12: # DC Mode
                    avg_ext = (top_hat[ikx,iky] + bot_hat[ikx,iky]) / 2.0
                    phi_p_int =  0.5 * Vm_hat[ikx,iky] + avg_ext
                    phi_m_int = -0.5 * Vm_hat[ikx,iky] + avg_ext
                else:
                    # STABLE LEAKAGE TERM:
                    # 1/cosh(x) = 2*exp(-x) / (1 + exp(-2x))
                    arg = lam * L
                    if arg > 700:
                        leakage_term = 0.0 # Electrode influence is zero at the membrane
                    else:
                        leakage_term = (top_hat[ikx,iky] + bot_hat[ikx,iky]) / (2.0 * np.cosh(arg))
                    
                    phi_p_int =  0.5 * Vm_hat[ikx,iky] + leakage_term
                    phi_m_int = -0.5 * Vm_hat[ikx,iky] + leakage_term

                # Populate Regions
                _solve_region_analytical(phi_hat, ikx, iky, lam, 0, self.k_mem_minus, 
                                          self.dz, bot_hat[ikx,iky], phi_m_int)
                _solve_region_analytical(phi_hat, ikx, iky, lam, self.k_mem_plus, self.Nz-1, 
                                          self.dz, phi_p_int, top_hat[ikx,iky])
                
                phi_hat[ikx, iky, self.Nz // 2] = (phi_p_int + phi_m_int) / 2.0

        for iz in range(self.Nz):
            self.phi[:, :, iz] = spfft.idctn(phi_hat[:, :, iz], type=2, norm='ortho')

        self.phi = self.comm.bcast(self.phi, root=0)
        return self

    def current_slice_kplus_minus(self, dz, sigma_e):
        """
        Compute J using second-order one-sided finite differences.
        O(dz²) accuracy at the membrane interface.
        """
        if self.rank != 0: return None
        
        k_plus = self.k_mem_plus
        k_minus = self.k_mem_minus
        
        # Second-order one-sided differences (forward at k_plus, backward at k_minus)
        # Forward:  f'(x) = (-3f(x) + 4f(x+h) - f(x+2h)) / (2h)
        # Backward: f'(x) = (3f(x) - 4f(x-h) + f(x-2h)) / (2h)
        
        grad_plus = (-3*self.phi[:,:,k_plus] + 4*self.phi[:,:,k_plus+1] - self.phi[:,:,k_plus+2]) / (2*dz)
        grad_minus = (3*self.phi[:,:,k_minus] - 4*self.phi[:,:,k_minus-1] + self.phi[:,:,k_minus-2]) / (2*dz)
        
        return 0.5 * sigma_e * (grad_plus + grad_minus)


@numba.njit(cache=True, fastmath=True)
def _solve_region_analytical(phi_hat, ikx, iky, lambda_k, z_start, z_end, dz, phi_left, phi_right):
    n_points = z_end - z_start + 1
    z_local = np.arange(n_points) * dz
    L_region = (n_points - 1) * dz
    sol_slice = phi_hat[ikx, iky, z_start : z_end + 1]

    if lambda_k < 1e-12:
        alpha = z_local / L_region
        sol_slice[:] = (1.0 - alpha) * phi_left + alpha * phi_right
        return

    arg_L = lambda_k * L_region
    if arg_L > 700: 
        term1 = phi_left * np.exp(-lambda_k * z_local)
        term2 = phi_right * np.exp(-lambda_k * (L_region - z_local))
        sol_slice[:] = term1 + term2
    else:
        denom = np.sinh(arg_L)
        term1 = phi_left * np.sinh(lambda_k * (L_region - z_local)) / denom
        term2 = phi_right * np.sinh(lambda_k * z_local) / denom
        sol_slice[:] = term1 + term2