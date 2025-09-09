# petsc_smoketest.py
from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

Nx, Ny, Nz = 96, 96, 97
N = Nx*Ny*Nz
dx = dy = dz = 1.0
cx, cy, cz = 1.0/dx**2, 1.0/dy**2, 1.0/dz**2

def gid(i,j,k): return i + Nx*(j + Ny*k)
def ijk(g):
    i = g % Nx; t = g // Nx
    j = t % Ny; k = t // Ny
    return i,j,k

# Matrix & vectors
A = PETSc.Mat().create(comm=comm)
A.setSizes(((PETSc.DECIDE, N), (PETSc.DECIDE, N)))
A.setType(PETSc.Mat.Type.MPIAIJ)
A.setPreallocationNNZ((7,7))
A.setOption(PETSc.Mat.Option.SYMMETRIC, True)
A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

b = PETSc.Vec().createMPI(N, comm=comm); b.set(0.0)
x = PETSc.Vec().createMPI(N, comm=comm); x.set(0.0)

r0, r1 = A.getOwnershipRange()
for row in range(r0, r1):
    i,j,k = ijk(row)

    # Interior stencil with Neumann mirror in x,y; z neighbors added here, ends handled by zeroRowsColumns
    diag = 0.0; cols = []; vals = []

    # x (Neumann mirror)
    if i == 0:
        cols.append(gid(i+1,j,k)); vals.append(-2.0*cx); diag += 2.0*cx
    elif i == Nx-1:
        cols.append(gid(i-1,j,k)); vals.append(-2.0*cx); diag += 2.0*cx
    else:
        cols.append(gid(i-1,j,k)); vals.append(-cx);     diag += cx
        cols.append(gid(i+1,j,k)); vals.append(-cx);     diag += cx

    # y (Neumann mirror)
    if j == 0:
        cols.append(gid(i,j+1,k)); vals.append(-2.0*cy); diag += 2.0*cy
    elif j == Ny-1:
        cols.append(gid(i,j-1,k)); vals.append(-2.0*cy); diag += 2.0*cy
    else:
        cols.append(gid(i,j-1,k)); vals.append(-cy);     diag += cy
        cols.append(gid(i,j+1,k)); vals.append(-cy);     diag += cy

    # z neighbors (both directions added; actual Dirichlet planes will be handled below)
    if k > 0:      cols.append(gid(i,j,k-1)); vals.append(-cz); diag += cz
    if k < Nz-1:   cols.append(gid(i,j,k+1)); vals.append(-cz); diag += cz

    cols.append(row); vals.append(diag)
    A.setValues([row], cols, vals)

A.assemblyBegin(); A.assemblyEnd()
b.assemblyBegin(); b.assemblyEnd()

# Dirichlet planes z=0 (phi=-0.5), z=Nz-1 (phi=+0.5), enforced by zeroRowsColumns
z0_rows = [gid(i,j,0)      for j in range(Ny) for i in range(Nx)]
z1_rows = [gid(i,j,Nz-1)   for j in range(Ny) for i in range(Nx)]
is0 = PETSc.IS().createGeneral(z0_rows, comm=comm)
is1 = PETSc.IS().createGeneral(z1_rows, comm=comm)

# Put boundary values into x; zero rows+cols with unit diag and adjust RHS consistently
for r in z0_rows: x.setValue(r, -0.5)
for r in z1_rows: x.setValue(r, +0.5)
x.assemble()

A.zeroRowsColumns(is0, 1.0, x, b)
A.zeroRowsColumns(is1, 1.0, x, b)

# Solve with GAMG
ksp = PETSc.KSP().create(comm=comm)
ksp.setOperators(A)
ksp.setType('cg')
pc = ksp.getPC(); pc.setType('gamg')
ksp.setTolerances(rtol=1e-6, atol=1e-50, max_it=200)
ksp.setFromOptions()

ksp.solve(b, x)
if rank == 0:
    print(f"[smoketest] iters={ksp.getIterationNumber()}, residual={ksp.getResidualNorm():.3e}")

# --- Validation ---

# 1) PETSc-reported convergence
reason = ksp.getConvergedReason()
if rank == 0:
    print("KSP reason:", reason)  # >0 means converged, <0 diverged, 0 max iters

# 2) Residual norms: r = b - A x
r = b.duplicate()
A.mult(x, r)         # r = A x
r.aypx(-1.0, b)      # r = -1*r + b = b - A x
rnorm_inf = r.norm(PETSc.NormType.NORM_INFINITY)
rnorm_2   = r.norm(PETSc.NormType.NORM_2)

bnorm_inf = b.norm(PETSc.NormType.NORM_INFINITY)
xnorm_inf = x.norm(PETSc.NormType.NORM_INFINITY)

if rank == 0:
    print(f"||b - A x||_inf = {rnorm_inf:.3e}  (rel to ||b||_inf: {rnorm_inf/max(bnorm_inf,1e-30):.3e})")
    print(f"||b - A x||_2   = {rnorm_2:.3e}  (rel to ||x||_inf: {rnorm_2/max(xnorm_inf,1e-30):.3e})")

# 3) Analytic check: phi_exact(k) = -0.5 + k/(Nz-1)
r0, r1 = A.getOwnershipRange()
local_max_err = 0.0
local_mean_err = 0.0
count = 0

arr = x.getArray(readonly=True)
for row in range(r0, r1):
    i = row % Nx
    t = row // Nx
    j = t % Ny
    k = t // Ny
    phi_exact = -0.5 + (k/(Nz-1))
    e = abs(arr[row - r0] - phi_exact)  # arr is local range
    local_max_err = max(local_max_err, e)
    local_mean_err += e
    count += 1

# reduce across ranks
max_err = comm.allreduce(local_max_err, op=MPI.MAX)
sum_err = comm.allreduce(local_mean_err, op=MPI.SUM)
sum_cnt = comm.allreduce(count, op=MPI.SUM)
mean_err = sum_err / max(sum_cnt, 1)

if rank == 0:
    print(f"Analytic check: max|phi-phi_exact| = {max_err:.3e}, mean|.| = {mean_err:.3e}")

# 4) Boundary plane means (should be ~-0.5 and +0.5)
# gather just first/last planes’ means
import numpy as np
local_z0_sum = 0.0; local_z1_sum = 0.0; nz0 = 0; nz1 = 0
for row in range(r0, r1):
    i = row % Nx
    t = row // Nx
    j = t % Ny
    k = t // Ny
    if k == 0:
        local_z0_sum += arr[row - r0]; nz0 += 1
    elif k == Nz-1:
        local_z1_sum += arr[row - r0]; nz1 += 1

z0_sum = comm.allreduce(local_z0_sum, op=MPI.SUM)
z1_sum = comm.allreduce(local_z1_sum, op=MPI.SUM)
z0_cnt = comm.allreduce(nz0, op=MPI.SUM)
z1_cnt = comm.allreduce(nz1, op=MPI.SUM)

if rank == 0:
    z0_mean = z0_sum / max(z0_cnt,1)
    z1_mean = z1_sum / max(z1_cnt,1)
    print(f"Boundary means: z=0  -> {z0_mean:.6f} (target -0.5),  z=Nz-1 -> {z1_mean:.6f} (target +0.5)")

