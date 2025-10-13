import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
from typing import List
from pycuda.compiler import SourceModule

from simulation_cpu import SimulationCPU   # if needed
from body import Body                      # particle class
from simulation_cuda import SimulationCUDA # base CUDA simulation
from quadtree_node import QuadTreeNode     # your quadtree node

class BarnesHutCUDA(SimulationCUDA):
    """
    Barnes-Hut accelerated N-body simulation using CUDA for updates.
    """

    def __init__(self, width: int, height: int, G: float, dt: float, n_bodies: int, theta: float = 0.5):
        # Initialize CUDA base
        super().__init__(width, height, G, dt, n_bodies)
        self.theta = theta

        # Root quadtree node
        self.root: Optional[QuadTreeNode] = None

        # Allocate force array on host and device
        self.forces = np.zeros((self.n_bodies, 2), dtype=np.float32)
        self.d_forces = cuda.mem_alloc(self.forces.nbytes)

        # CUDA kernel for updating positions
        cuda_code = """
        extern "C" __global__ void update_positions(float* positions, float* velocities, float* forces,
                                                    float* masses, int n, float dt) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;

            float fx = forces[2*idx];
            float fy = forces[2*idx+1];
            float mass = masses[idx];

            float ax = fx / (mass + 1e-8f);
            float ay = fy / (mass + 1e-8f);

            velocities[2*idx] += ax * dt;
            velocities[2*idx+1] += ay * dt;

            positions[2*idx] += velocities[2*idx] * dt;
            positions[2*idx+1] += velocities[2*idx+1] * dt;
        }
        """
        self.mod2 = SourceModule(cuda_code)
        self.update_kernel = self.mod2.get_function("update_positions")

    # ----------------------------
    # Build quadtree
    # ----------------------------
    def build_tree(self):
        """Build Barnes-Hut quadtree from current bodies."""
        self.root = QuadTreeNode(self.width / 2, self.height / 2, max(self.width, self.height))
        for body in self.bodies:
            self.root.insert(body)

    # ----------------------------
    # Compute forces
    # ----------------------------
    def compute_forces(self):
        """Compute forces on all bodies using the Barnes-Hut quadtree."""
        if self.root is None:
            raise RuntimeError("Quadtree not built. Call build_tree() first.")

        for i, body in enumerate(self.bodies):
            fx, fy = self.root.calculate_force(body, self.theta)
            self.forces[i, 0] = fx
            self.forces[i, 1] = fy

    # ----------------------------
    # Update simulation
    # ----------------------------
    def update(self):
        start = time.perf_counter()

        # 1. Build quadtree
        self.build_tree()

        # 2. Compute forces
        self.compute_forces()

        # 3. Copy forces to GPU
        cuda.memcpy_htod(self.d_forces, self.forces)

        # 4. Update positions & velocities on GPU
        block_size = min(256, self.n_bodies)
        grid_size = (self.n_bodies + block_size - 1) // block_size
        self.update_kernel(
            self.d_positions, self.d_velocities, self.d_forces,
            self.d_masses, np.int32(self.n_bodies), np.float32(self.dt),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )

        # 5. Copy updated positions back
        cuda.memcpy_dtoh(self.positions_flat, self.d_positions)
        self.positions = self.positions_flat.reshape((self.n_bodies, 2))

        # 6. Update Body objects
        for i, body in enumerate(self.bodies):
            body.x, body.y = self.positions[i]

        # Record timing
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.performance_stats[self.stat_index] = elapsed_ms
        self.stat_index = (self.stat_index + 1) % len(self.performance_stats)

    # ----------------------------
    # Reset simulation
    # ----------------------------
    def reset(self):
        """Reset bodies, forces, and quadtree."""
        super().reset()
        self.forces.fill(0.0)
        cuda.memcpy_htod(self.d_forces, self.forces)
        self.root = None
