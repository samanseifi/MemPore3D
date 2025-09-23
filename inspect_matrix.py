from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD

# Create a dummy DMDA and Matrix, just like in the main script
da = PETSc.DMDA().create([10, 10, 10], comm=comm)
A = da.createMatrix()

if comm.Get_rank() == 0:
    print("--- Inspecting the PETSc Matrix Object ---")
    print("Object type:", type(A))
    print("\n--- Available methods and attributes (looking for 'setValuesStencil') ---")
    
    found = False
    # Print all attributes that don't start with a double underscore
    for method_name in dir(A):
        if not method_name.startswith('__'):
            print(method_name)
            if method_name == 'setValuesStencil':
                found = True

    print("-----------------------------------------")
    if found:
        print("\nSUCCESS: 'setValuesStencil' was found in the list of methods.")
    else:
        print("\nFAILURE: 'setValuesStencil' was NOT found. This is highly unexpected.")
    print("-----------------------------------------")
