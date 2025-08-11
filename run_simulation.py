import json
import argparse
from visualization import AllenCahnVisualization
from solver.allen_cahn_solver import AllenCahnSolver

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run the Allen-Cahn simulation.')
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration JSON file.')
    parser.add_argument('--no-viz', action='store_true', help='Run simulation without visualization.')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save output data.')
    parser.add_argument('--load-data', type=str, help='Load simulation data from a file for visualization.')
    parser.add_argument('--plots', nargs='+', help='List of plots to display: phi, color_gradient, area_deviation, sigma.')
    args = parser.parse_args()

    # Load simulation parameters from JSON file
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Initialize the solver with parameters from the config
    solver = AllenCahnSolver(
        L_tilde=config.get('L_tilde', 1000.0),
        N=config.get('N', 5000),
        epsilon=config.get('epsilon', 1.0),
        sigma_initial=config.get('sigma_initial', 0.0),
        sigma_final=config.get('sigma_final', 1.0),
        noise_strength=config.get('noise_strength', 0.001),
        kappa=config.get('kappa', 0.1),
        K_A=config.get('K_A', 0.1),
        phi0=config.get('phi0', 1.0),
        t_total=config.get('t_total', 1.0)
    )

    T_steps = config.get('T_steps', 5000)
    save_interval = config.get('save_interval', 100)

    if args.no_viz:
        # Run the simulation without visualization
        solver.run(T_steps, save_interval=save_interval, output_dir=args.output_dir)
    elif args.load_data:
        # Visualize using precomputed data
        vis = AllenCahnVisualization(solver, plots_to_show=args.plots)
        vis.animate_from_saved_data(args.load_data)
    else:
        # Run the simulation with visualization using the run functionality
        vis = AllenCahnVisualization(solver, plots_to_show=args.plots)
        vis.run_and_visualize(T_steps)

if __name__ == "__main__":
    main()
