import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import os
import re
import argparse

# --- 1. Style & Utility Setup ---
def setup_matplotlib_for_latex():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "axes.linewidth": 0.6,
        "xtick.direction": 'in',
        "ytick.direction": 'in',
    })

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def calculate_pore_stats(psi, x, y):
    dx, dy = x[1] - x[0], y[1] - y[0]
    pore_area = np.sum(psi < 0.5) * dx * dy
    return np.sqrt(pore_area / np.pi) if pore_area > 0 else 0.0

def load_simulation_files(results_dir):
    file_pattern = os.path.join(results_dir, "step_*.npz")
    files = glob.glob(file_pattern)
    files.sort(key=natural_sort_key)
    if not files:
        raise FileNotFoundError(f"No files found in '{results_dir}'")
    return files

def create_analysis_video(results_dir, output_file="analysis_evolution.mp4", 
                          fps=15, dpi=150, shift_x=0, shift_y=0):
    setup_matplotlib_for_latex()
    files = load_simulation_files(results_dir)

    time_hist, radius_hist, vm_hist = [], [], []

    # Setup Figure: [2D Map] [Combined Line Plot]
    fig, (ax_map, ax_lines) = plt.subplots(1, 2, figsize=(10, 4))
    
    with np.load(files[0], allow_pickle=True) as data:
        x_um, y_um = data['x'] * 1e6, data['y'] * 1e6
        psi = data['psi']
    
    # Pre-calculate shift indices
    shift_pts = (shift_x, shift_y)
    
    # Left Panel: Phase Field + Initial Contour
    psi_vis = np.roll(psi, shift_pts, axis=(0,1))
    im = ax_map.imshow(psi_vis.T, origin="lower", 
                       extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]], 
                       cmap="pink_r", vmin=0, vmax=1)
    
    # Draw initial pore border (contour level 0.5)
    ax_map.contour(x_um, y_um, psi_vis.T, levels=[0.5], 
                   colors='gray', linewidths=0.8, linestyles='--')
    
    ax_map.set_xlabel(r"$x$ ($\mu$m)")
    ax_map.set_ylabel(r"$y$ ($\mu$m)")
    ax_map.set_title(r"Phase Field $\phi$")

    # Right Panel: Combined Vm and Radius
    ax_vm = ax_lines
    ax_rad = ax_vm.twinx()
    
    # --- FIXED AXES MODIFICATIONS ---
    ax_vm.set_xlim(0, 50)     # Fixed X at 50 us
    ax_rad.set_ylim(0, 9)     # Fixed Y at 9 nm for radius
    # ----------------------------------

    line_vm, = ax_vm.plot([], [], color='#1b9e77', label=r"Avg $V_m$")
    line_r, = ax_rad.plot([], [], color='#d95f02', label=r"$R_{pore}$")
    
    ax_vm.set_xlabel(r"Time ($\mu$s)")
    ax_vm.set_ylabel(r"Voltage $V_m$ (V)", color='#1b9e77')
    ax_rad.set_ylabel(r"Radius $R_{pore}$ (nm)", color='#d95f02')
    
    lns = [line_vm, line_r]
    labs = [l.get_label() for l in lns]
    ax_vm.legend(lns, labs, loc='upper left', frameon=True, framealpha=0.8)

    def update(frame_idx):
        with np.load(files[frame_idx], allow_pickle=True) as data:
            t = float(data['time']) * 1e6
            psi = data['psi']
            x, y = data['x'], data['y']
            
            vm_val = data['avg_Vm'] if 'avg_Vm' in data else np.mean(data['Vm'])
            r_nm = calculate_pore_stats(psi, x, y) * 1e9
            
            time_hist.append(t)
            vm_hist.append(vm_val)
            radius_hist.append(r_nm)

            # 1. Update 2D Map Array
            psi_vis = np.roll(psi, shift_pts, axis=(0,1))
            im.set_array(psi_vis.T)
            
            # 2. Update Pore Border (Clear old contours, draw new)
            for c in ax_map.collections:
                if c is not im: # Don't remove the main image
                    c.remove()
            ax_map.contour(x_um, y_um, psi_vis.T, levels=[0.5], 
                           colors='gray', linewidths=0.8, linestyles='--')
            
            # 3. Update Line Plots
            line_vm.set_data(time_hist, vm_hist)
            line_r.set_data(time_hist, radius_hist)
            
            # 4. Dynamic Y-scaling for Vm only (X and Radius are fixed)
            ax_vm.relim()
            ax_vm.autoscale_view(scalex=False, scaley=True)

            if frame_idx % 20 == 0:
                print(f"Processing frame {frame_idx}/{len(files)}")

        return [im, line_vm, line_r]

    ani = FuncAnimation(fig, update, frames=len(files), blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=3000)
    ani.save(output_file, writer=writer, dpi=dpi)
    plt.close()
    print(f"Analysis video saved to: {output_file}")

# --- 3. Simple Field Video (Original Request with Shifts) ---
def create_phase_field_video(results_dir, output_file, fps, dpi, field, sx, sy):
    setup_matplotlib_for_latex()
    files = load_simulation_files(results_dir)

    with np.load(files[0], allow_pickle=True) as data:
        x, y = data['x'] * 1e6, data['y'] * 1e6
        first_frame = np.roll(data[field], (sx, sy), axis=(0, 1))

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(first_frame.T, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap='pink_r')
    
    def update(i):
        with np.load(files[i], allow_pickle=True) as data:
            field_data = np.roll(data[field], (sx, sy), axis=(0, 1))
            im.set_array(field_data.T)
        return [im]

    ani = FuncAnimation(fig, update, frames=len(files), blit=False)
    ani.save(output_file, writer=FFMpegWriter(fps=fps), dpi=dpi)
    plt.close()

# --- 4. Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", nargs='?', default="simulation_results")
    parser.add_argument("-o", "--output", default="evolution_analysis.mp4")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--field", choices=['psi', 'Vm', 'phi_elec'], default='psi')
    parser.add_argument("--dual", action="store_true", help="Create analysis video (Map + Vm/Radius plot)")
    parser.add_argument("--sx", type=int, default=0, help="Shift in X (indices)")
    parser.add_argument("--sy", type=int, default=0, help="Shift in Y (indices)")

    args = parser.parse_args()

    if args.dual:
        create_analysis_video(args.results_dir, args.output, args.fps, args.dpi, args.sx, args.sy)
    else:
        create_phase_field_video(args.results_dir, args.output, args.fps, args.dpi, args.field, args.sx, args.sy)