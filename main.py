from mpi4py import MPI
import time, sys
import matplotlib.pyplot as plt
from body import WIDTH, HEIGHT, G, DT, N_BODIES
from simulation_cpu import SimulationCPU
from simulation_cuda import SimulationCUDA
from ui import Visualization

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ------------------- Helpers -------------------

def compute_metrics(bodies, G):
    import numpy as np
    positions = np.array([[b.x, b.y] for b in bodies])
    velocities = np.array([[b.vx, b.vy] for b in bodies])
    masses = np.array([b.mass for b in bodies])
    v = np.linalg.norm(velocities, axis=1)
    KE = 0.5 * np.sum(masses * v**2)
    PE = 0.0
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            r = np.linalg.norm(positions[i] - positions[j])
            if r > 1e-6:
                PE -= G * masses[i] * masses[j] / r
    total_E = KE + PE
    momentum = np.linalg.norm(np.sum(masses[:, None] * velocities, axis=0))
    return {
        "speed_avg": float(np.mean(v)),
        "energy": float(total_E),
        "momentum": float(momentum)
    }

# ------------------- Simulation Function -------------------

def run_simulation(sim_class, n_bodies=N_BODIES, steps=5000):
    sim = sim_class(WIDTH, HEIGHT, G, DT, n_bodies)
    step_times = []
    for _ in range(steps):
        t0 = time.time()
        sim.update()
        step_time = (time.time() - t0) * 1000  # ms
        step_times.append(step_time)
        metrics = compute_metrics(sim.bodies, G)
        comm.send((sim.bodies, step_time, metrics), dest=0)
    return step_times

# ------------------- Main -------------------

def main():
    if size < 2:
        if rank == 0:
            print("Run with at least 2 MPI processes for visualization: mpirun -n 2 python main.py")
        sys.exit(1)

    if rank == 0:
        # ---------- Visualization Master ----------
        visualization = Visualization()
        fps = 0
        last_stat_time = 0
        running = True

        # collect performance data from rank 1 after runs
        cpu_times = None
        cuda_times = None

        while running:
            data = comm.recv(source=1)
            if isinstance(data, tuple) and data[0] is None:   # sentinel exit
                break
            if isinstance(data, dict) and "cpu_times" in data:
                cpu_times = data["cpu_times"]
                cuda_times = data["cuda_times"]
                continue

            bodies_list, step_time, metrics = data
            now = time.time()
            if now - last_stat_time > 0.1:
                fps = visualization.get_fps()
                last_stat_time = now

            visualization.render(bodies_list, step_time, fps)
            running, reset_needed = visualization.handle_events()
            visualization.tick()

        visualization.quit()

        # ---------- Show Performance Plot ----------
        if cpu_times and cuda_times:
            plt.figure(figsize=(8, 5))
            plt.plot(cpu_times, label="CPU", color='blue')
            plt.plot(cuda_times, label="CUDA", color='red')
            plt.title("Step Time per Iteration")
            plt.xlabel("Step")
            plt.ylabel("Time (ms)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("performance_comparison.png")

    elif rank == 1:
        # ---------- Simulation Worker ----------
        print("Running CPU simulation...")
        cpu_times = run_simulation(SimulationCPU, steps=500)
        print("CPU simulation done!")

        print("Running CUDA simulation...")
        cuda_times = run_simulation(SimulationCUDA, steps=500)
        print("CUDA simulation done!")

        # Send times for plotting
        comm.send({"cpu_times": cpu_times, "cuda_times": cuda_times}, dest=0)

        # Signal visualization to exit
        comm.send((None, None, None), dest=0)

if __name__ == "__main__":
    main()
