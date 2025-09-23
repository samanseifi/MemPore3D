from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# --- Problem Parameters ---
Nx, Ny, Nz = 96, 96, 97
dx = dy = dz = 1.0
dx2, dy2, dz2 = dx*dx, dy*dy, dz*dz
cx, cy, cz = 1.0/dx2, 1.0/dy2, 1.0/dz2
diag = 2.0 * (cx + cy + cz)

# --- PETSc DMDA Setup ---
da = PETSc.DMDA().create(
    [Nx, Ny, Nz], dof=1, stencil_width=1,
    boundary_type=(PETSc.DM.BoundaryType.PERIODIC,
                   PETSc.DM.BoundaryType.PERIODIC,
                   PETSc.DM.BoundaryType.NONE),
    stencil_type=PETSc.DMDA.StencilType.BOX,
    comm=comm
)

# --- Matrix and Vector Creation ---
A = da.createMatrix()
b = da.createGlobalVec()
x = da.createGlobalVec()
b.set(0.0)

# --- Matrix Assembly ---
(xs, xe), (ys, ye), (zs, ze) = da.getRanges()
row = PETSc.Mat.Stencil()

for k in range(zs, ze):
    for j in range(ys, ye):
        for i in range(xs, xe):
            row.index = (i, j, k)
            row_gid = k * Nx * Ny + j * Nx + i

            # For setting single entries, setValueStencil is correct and efficient.
            if k == 0:
                A.setValueStencil(row, row, 1.0)
                b.setValue(row_gid, -0.5)
            elif k == Nz - 1:
                A.setValueStencil(row, row, 1.0)
                b.setValue(row_gid, +0.5)
            else:
                # For setting a full stencil row, we use setValues.
                cols_indices = [
                    (i, j, k), (i-1, j, k), (i+1, j, k), (i, j-1, k),
                    (i, j+1, k), (i, j, k-1), (i, j, k+1)
                ]
                data = [diag, -cx, -cx, -cy, -cy, -cz, -cz]
                
                # THIS IS THE FINAL, CORRECT API CALL.
                A.setValues([row], cols_indices, data)

A.assemble()
b.assemble()

# --- KSP Solver Setup and Solve ---
ksp = PETSc.KSP().create(comm=comm)
ksp.setOperators(A)
ksp.setType('cg')
pc = ksp.getPC()
pc.setType('gamg')
ksp.setTolerances(rtol=1e-6, atol=1e-50, max_it=200)
ksp.setFromOptions()
ksp.solve(b, x)

# --- Output Results ---
its = ksp.getIterationNumber()
res = ksp.getResidualNorm()
if rank == 0:
    print(f"[{PETSc.Sys.getDate()}] iters={its}, residual={res:.3e}  (OK if ~1e-6)")
