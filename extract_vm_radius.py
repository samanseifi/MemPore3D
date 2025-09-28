"""
Extracts simulation data from .npz files into human-readable CSV files.

This script processes a list of .npz files. For each input file, it
creates two output .csv files:
1.  One for the membrane potential (Vm) vs. time.
2.  One for the pore radius vs. time.

Usage:
    python extract_to_csv.py
    (Modify the 'npz_files' list in the script to point to your files)
"""
import os
import numpy as np
from pathlib import Path

def extract_data_to_csv(filepaths: list):
    """
    Loads data from a list of .npz files and saves Vm and radius data to CSVs.

    Args:
        filepaths (list): A list of strings, where each is a path to an .npz file.
    """
    print("🚀 Starting data extraction to CSV...")

    # Loop through each file provided in the list
    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"❌ Warning: File not found at '{filepath}'. Skipping.")
            continue

        try:
            print(f"\nProcessing '{filepath}'...")
            with np.load(filepath, allow_pickle=True) as data:
                # --- 1. Load the core arrays ---
                time_s = data['time_points']
                avg_Vm_vs_time = data['avg_Vm_vs_time']
                
                # Use .get() for radius, as it might not be in every file
                pore_radius_m = data.get('pore_radius_vs_time', None)

                # Get the base name of the file for naming the outputs
                # e.g., "results_run_1.npz" -> "results_run_1"
                base_name = Path(filepath).stem

                # --- 2. Save Vm data to its CSV file ---
                output_vm_path = f"{base_name}_vm.csv"
                
                # Stack the two 1D arrays (time, Vm) into a 2D array for saving
                vm_data_to_save = np.stack((time_s, avg_Vm_vs_time), axis=1)
                
                # Use numpy.savetxt for easy CSV creation
                np.savetxt(
                    output_vm_path,
                    vm_data_to_save,
                    delimiter=",",
                    header="Time (s),Vm (V)",
                    comments="" # Removes the '#' from the header line
                )
                print(f"  ✅ Saved Vm data to '{output_vm_path}'")

                # --- 3. Save radius data if it exists ---
                if pore_radius_m is not None:
                    output_radius_path = f"{base_name}_radius.csv"
                    
                    # Stack the two 1D arrays (time, radius) into a 2D array
                    radius_data_to_save = np.stack((time_s, pore_radius_m), axis=1)
                    
                    np.savetxt(
                        output_radius_path,
                        radius_data_to_save,
                        delimiter=",",
                        header="Time (s),Radius (m)",
                        comments=""
                    )
                    print(f"  ✅ Saved radius data to '{output_radius_path}'")
                else:
                    print("  -> No radius data found in this file. Skipping radius CSV.")

        except Exception as e:
            print(f"❌ An error occurred while processing {filepath}: {e}")

    print("\n🎉 Extraction complete.")


if __name__ == "__main__":
    # --- IMPORTANT ---
    # List the paths to your .npz files here.
    npz_files = [
        "simulation_results_low_field.npz",
        "simulation_results_high_field.npz",
        "simulation_no_radius.npz" # A file we'll create without radius data
    ]

    # Create dummy .npz files for testing if they don't exist
    for f in npz_files:
        if not os.path.exists(f):
            print(f"Creating dummy file for demonstration: {f}")
            time = np.linspace(0, 5e-9, 100)
            
            # Differentiate the dummy data for visual distinction
            if "high" in f:
                vm = 1.2 * (1 - np.exp(-time / 1e-9))
                radius = 0.5e-9 + 2.5e-9 * (1 - np.exp(-time / 2e-9))
                np.savez(f, time_points=time, avg_Vm_vs_time=vm, pore_radius_vs_time=radius)
            elif "low" in f:
                vm = 0.8 * (1 - np.exp(-time / 1.5e-9))
                radius = 0.5e-9 + 1.0e-9 * (1 - np.exp(-time / 3e-9))
                np.savez(f, time_points=time, avg_Vm_vs_time=vm, pore_radius_vs_time=radius)
            else: # Create a file without radius data
                vm = 0.5 * (1 - np.exp(-time / 2e-9))
                np.savez(f, time_points=time, avg_Vm_vs_time=vm)
    
    # Run the main extraction function
    extract_data_to_csv(npz_files)