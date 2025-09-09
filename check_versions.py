import sys
import petsc4py
from petsc4py import PETSc

print("--- PETSc Version Check ---")
print("Python Executable:", sys.executable)
print("petsc4py version:   ", petsc4py.__version__)
print("petsc4py is linked to PETSc version:", PETSc.Sys.getVersion())
print("---------------------------")
