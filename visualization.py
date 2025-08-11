import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Import the solver
from solver.allen_cahn_solver import AllenCahnSolver

def heaviside(phi, k):
    """Approximation of the Heaviside function."""
    return 0.5 * (1 + np.tanh(k * (phi - 0.5)))

class AllenCahnVisualization:
    """Class for visualizing the Allen-Cahn simulation."""

    def __init__(self, solver, plots_to_show=None):
        """
        Initialize the visualization.

        Parameters:
        - solver: An instance of AllenCahnSolver.
        - plots_to_show: A list of plots to display. Options are:
          'phi', 'color_gradient', 'area_deviation', 'sigma'. If None, 'phi' and 'color_gradient' are shown by default.
        """
        self.solver = solver
        self.plots_to_show = plots_to_show or ['phi', 'color_gradient']

        self.x = np.linspace(0, self.solver.L_tilde, self.solver.N)

        # Determine the number of subplots based on plots_to_show
        self.num_plots = len(self.plots_to_show)
        self.fig, self.axes = plt.subplots(self.num_plots, 1, figsize=(8, 4 * self.num_plots))

        # If only one plot, make self.axes a list for consistency
        if self.num_plots == 1:
            self.axes = [self.axes]

        self.plot_indices = {}  # Map plot names to subplot indices
        idx = 0

        # Initialize plots based on plots_to_show
        if 'phi' in self.plots_to_show:
            self.init_phi_plot(self.axes[idx])
            self.plot_indices['phi'] = idx
            idx += 1

        if 'color_gradient' in self.plots_to_show:
            self.init_color_gradient_plot(self.axes[idx])
            self.plot_indices['color_gradient'] = idx
            idx += 1

        if 'area_deviation' in self.plots_to_show:
            self.init_area_deviation_plot(self.axes[idx])
            self.plot_indices['area_deviation'] = idx
            idx += 1

        if 'sigma' in self.plots_to_show:
            self.init_sigma_plot(self.axes[idx])
            self.plot_indices['sigma'] = idx

    def init_phi_plot(self, ax):
        """Initialize the φ (order parameter) plot."""
        self.line_phi, = ax.plot(self.x, self.solver.phi)
        ax.set_ylim([0, 1.5])
        ax.set_xlim([0, self.solver.L_tilde])
        ax.set_xlabel('Nondimensional Position (x)')
        ax.set_ylabel('Order Parameter (φ)')
        ax.set_title('Order Parameter Over Time')
        self.time_text_phi = ax.text(0.8 * self.solver.L_tilde, 1.3, '', fontsize=12)

    def init_color_gradient_plot(self, ax):
        """Initialize the color gradient visualization plot."""
        self.lc = self.get_colored_line(self.solver.phi)
        ax.add_collection(self.lc)
        ax.set_ylim([-0.05, 0.05])
        ax.set_xlim([0, self.solver.L_tilde])
        ax.set_xlabel('Nondimensional Position (x)')
        ax.set_ylabel('Order Parameter (φ)')
        ax.set_title('Color Gradient Visualization')
        self.time_text_color = ax.text(0.8 * self.solver.L_tilde, 0.03, '', fontsize=12)

        # Colorbar setup
        sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        self.cbar = plt.colorbar(sm, ax=ax)
        self.cbar.set_label('Order Parameter (φ)')

    def init_area_deviation_plot(self, ax):
        """Initialize the area deviation plot."""
        self.area_deviation_list = []
        self.time_data = []
        self.line_area, = ax.plot([], [], 'r-')
        ax.set_xlim(0, 1)  # Will be updated dynamically
        ax.set_xlabel('Time')
        ax.set_ylabel('Area Deviation')
        ax.set_title('Area Deviation Over Time')

    def init_sigma_plot(self, ax):
        """Initialize the sigma (surface tension) plot."""
        self.sigma_list = []
        self.time_data = []
        self.line_sigma, = ax.plot([], [], 'g-')
        ax.set_xlim(0, 1)  # Will be updated dynamically
        ax.set_xlabel('Time')
        ax.set_ylabel('Surface Tension (σ)')
        ax.set_title('Surface Tension Over Time')

    def get_colored_line(self, phi):
        """Create a color-mapped line based on φ values."""
        colors = cm.jet(Normalize(vmin=0, vmax=1)(phi[:-1]))
        segments = np.array([self.x[:-1], self.x[1:]]).T.reshape(-1, 2)
        lc = LineCollection(
            [[(segments[i, 0], 0), (segments[i, 1], 0)] for i in range(len(segments))],
            colors=colors,
            linewidth=3
        )
        return lc

    def update_visualization(self, step):
        """Update the plots with the current state of the solver."""
        if 'phi' in self.plots_to_show:
            idx = self.plot_indices['phi']
            self.line_phi.set_ydata(self.solver.phi)
            self.time_text_phi.set_text(f'Time: {self.solver.time:.4e}')

        if 'color_gradient' in self.plots_to_show:
            idx = self.plot_indices['color_gradient']
            self.lc.remove()
            self.lc = self.get_colored_line(self.solver.phi)
            self.axes[idx].add_collection(self.lc)
            self.time_text_color.set_text(f'Time: {self.solver.time:.4e}')

        if 'area_deviation' in self.plots_to_show:
            idx = self.plot_indices['area_deviation']
            H = heaviside(self.solver.phi, k=10.0)
            current_area = np.sum(H) * self.solver.dx_tilde
            area_deviation = current_area - self.solver.A0
            self.area_deviation_list.append(area_deviation)
            self.time_data.append(self.solver.time)
            self.line_area.set_data(self.time_data, self.area_deviation_list)
            self.axes[idx].set_xlim(0, self.solver.time)
            self.axes[idx].set_ylim(min(self.area_deviation_list), max(self.area_deviation_list))

        if 'sigma' in self.plots_to_show:
            idx = self.plot_indices['sigma']
            self.sigma_list.append(self.solver.sigma)
            self.time_data.append(self.solver.time)
            self.line_sigma.set_data(self.time_data, self.sigma_list)
            self.axes[idx].set_xlim(0, self.solver.time)
            self.axes[idx].set_ylim(min(self.sigma_list), max(self.sigma_list))

        plt.pause(0.001)  # Small pause to allow the plot to update

    def run_and_visualize(self, T_steps):
        """Run the simulation and update visualization."""
        # Start interactive mode
        plt.ion()
        plt.tight_layout()
        plt.show()

        # Run the simulation with the update_visualization as the callback
        self.solver.run(T_steps, callback=self.update_visualization)

        # End interactive mode
        plt.ioff()
        plt.show()
