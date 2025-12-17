import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re

# --- 1. Safe Matplotlib Settings (No external LaTeX required) ---
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",  # Computer Modern look-alike
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,         # Good resolution for screen
})

def natural_sort_key(s):
    """Sorts strings containing numbers naturally (e.g., step_2 before step_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def load_data(results_dir="simulation_results"):
    """Reads all step_*.npz files and extracts history and final state."""
    
    # Find all step files
    file_pattern = os.path.join(results_dir, "step_*.npz")
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"❌ No files found in '{results_dir}'")
        return None, None

    # Sort files numerically
    files.sort(key=natural_sort_key)
    print(f"✅ Found {len(files)} time steps. Loading data...")

    # Lists to store history
    times = []
    radii = []

    # Loop through all files to build history
    for f in files:
        with np.load(f, allow_pickle=True) as data:
            times.append(data['time'])
            radii.append(data['pore_radius'])
    
    # Convert to arrays
    times = np.array(times)
    radii = np.array(radii)

    # Load full 2D data from the LAST file for the contour plot
    last_file = files[-1]
    print(f"   -> Loading final state from: {os.path.basename(last_file)}")
    with np.load(last_file, allow_pickle=True) as data:
        final_state = {
            'x': data['x'],
            'y': data['y'],
            'psi': data['psi'],
            'time': data['time']
        }

    return (times, radii), final_state

def plot_pore_radius(times, radii):
    """Plots Pore Radius vs. Time."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    # Convert to convenient units (ns and nm)
    t_ns = times * 1e9
    r_nm = radii * 1e9

    ax.plot(t_ns, r_nm, '-', color='#d95f02', linewidth=1.5, label='Pore Radius')
    
    ax.set_xlabel(r"Time ($t$) [ns]")
    ax.set_ylabel(r"Effective Radius ($R$) [nm]")
    ax.set_title("Pore Growth Dynamics")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig("plot_pore_radius.png", dpi=300)
    print("   -> Saved 'plot_pore_radius.png'")
    plt.show()

def plot_final_contour(final_state):
    """Plots the Phase Field contour of the final state."""
    x = final_state['x'] * 1e9  # Convert to nm
    y = final_state['y'] * 1e9
    psi = final_state['psi']
    t_final = final_state['time'] * 1e9

    fig, ax = plt.subplots(figsize=(5, 4.5))
    
    # Plot heatmap
    # Transpose psi.T often needed because meshgrid indexing (ij vs xy)
    im = ax.imshow(psi.T, origin='lower', 
                   extent=[x[0], x[-1], y[0], y[-1]], 
                   cmap='Blues_r', vmin=0, vmax=1)
    
    # Add a contour line at psi=0.5 (the interface)
    ax.contour(x, y, psi.T, levels=[0.5], colors='k', linewidths=1.0, linestyles='--')

    ax.set_xlabel(r"$x$ [nm]")
    ax.set_ylabel(r"$y$ [nm]")
    ax.set_title(f"Final Phase Field $\phi$ (t = {t_final:.1f} ns)")
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(r"Phase Field value")
    
    plt.tight_layout()
    plt.savefig("plot_final_phase.png", dpi=300)
    print("   -> Saved 'plot_final_phase.png'")
    plt.show()

if __name__ == "__main__":
    # 1. Load Data
    history, final_state = load_data()

    if history is not None:
        times, radii = history
        
        # 2. Plot Radius
        plot_pore_radius(times, radii)
        
        # 3. Plot Final Contour
        plot_final_contour(final_state)
        
        print("\n✨ Done.")