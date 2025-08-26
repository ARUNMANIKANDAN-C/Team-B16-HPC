import time
import sys
from body import WIDTH, HEIGHT, G, DT, N_BODIES  # Import constants from Part 1
from simulation_cuda import SimulationCUDA  # Import from Part 2
from ui import Visualization  # Import from Part 3
def main():
    """Main function to run the N-Body simulation."""
    # Initialize CUDA simulation and visualization
    simulation = SimulationCUDA(WIDTH, HEIGHT, G, DT, N_BODIES)
    visualization = Visualization()
    
    running = True
    last_stat_time = 0
    avg_time = 0
    fps = 0
    
    while running:
        # Handle events
        running, reset_needed = visualization.handle_events()
        
        if reset_needed:
            simulation.reset()
        
        # Update simulation
        simulation.update()
        
        # Update stats less frequently
        current_time = time.time()
        if current_time - last_stat_time > 0.1:  # Update stats every 100ms
            avg_time = simulation.get_avg_time()
            fps = visualization.get_fps()
            last_stat_time = current_time
        
        # Render the simulation
        visualization.render(simulation.bodies, avg_time, fps)
        visualization.tick()
    
    visualization.quit()
    sys.exit()

if __name__ == "__main__":
    main()
