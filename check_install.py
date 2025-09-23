# check_install.py

import mpi4py_fft
import sys

print("--- Checking mpi4py-fft Installation ---")
print(f"Python Executable: {sys.executable}")
print("-" * 40)
try:
    print(f"Found mpi4py-fft version: {mpi4py_fft.__version__}")
    print(f"Found at file path:      {mpi4py_fft.__file__}")
    print("-" * 40)

    # Now, let's try the import that was failing
    from mpi4py_fft import arrays
    print("SUCCESS: The 'arrays' submodule was imported correctly.")

except Exception as e:
    print(f"FAILURE: An error occurred.")
    print(f"Error details: {e}")

print("-" * 40)