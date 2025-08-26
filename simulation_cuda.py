import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from typing import List
from body import Body  # Import Body from Part 1
import time
class SimulationCUDA:
    """CUDA-based implementation of the N-Body simulation."""
    
    def __init__(self, width: int, height: int, G: float, dt: float, n_bodies: int):
        self.width = width
        self.height = height
        self.G = G
        self.dt = dt
        self.n_bodies = n_bodies
        self.bodies = self._create_bodies()
        self.performance_stats = np.zeros(100, dtype=np.float32)
        self.stat_index = 0
        self._setup_cuda()
    
    def _create_bodies(self) -> List[Body]:
        """Create initial set of bodies."""
        return [Body.create_random(self.width, self.height) for _ in range(self.n_bodies)]
    
    def _setup_cuda(self) -> None:
        """Initialize CUDA resources."""
        # Convert bodies to numpy arrays (pre-allocate)
        self.positions = np.zeros((self.n_bodies, 2), dtype=np.float32)
        self.velocities = np.zeros((self.n_bodies, 2), dtype=np.float32)
        self.masses = np.zeros(self.n_bodies, dtype=np.float32)
        
        # Initialize arrays with body data
        for i, body in enumerate(self.bodies):
            self.positions[i] = [body.x, body.y]
            self.velocities[i] = [body.vx, body.vy]
            self.masses[i] = body.mass
        
        # Flatten positions and velocities for CUDA
        self.positions_flat = self.positions.flatten()
        self.velocities_flat = self.velocities.flatten()
        
        # CUDA kernel with extern "C" for proper linkage
        cuda_code = """
        #define SOFTENING 1e-3f
        
        extern "C" __global__ void nbody_update(float* positions, float* velocities, float* masses, 
                                             int num_bodies, float G, float dt) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= num_bodies) return;
            
            float ax = 0.0f;
            float ay = 0.0f;
            
            float my_x = positions[2*idx];
            float my_y = positions[2*idx+1];
            float my_mass = masses[idx];
            
            // Pre-load other bodies into shared memory for better performance
            __shared__ float shared_pos[256 * 2];
            __shared__ float shared_mass[256];
            
            for (int tile = 0; tile < gridDim.x; tile++) {
                int tid = threadIdx.x;
                int load_idx = tile * blockDim.x + tid;
                
                if (load_idx < num_bodies) {
                    shared_pos[2*tid] = positions[2*load_idx];
                    shared_pos[2*tid+1] = positions[2*load_idx+1];
                    shared_mass[tid] = masses[load_idx];
                }
                __syncthreads();
                
                // Calculate forces from bodies in shared memory
                for (int j = 0; j < blockDim.x; j++) {
                    int body_idx = tile * blockDim.x + j;
                    if (body_idx >= num_bodies) break;
                    if (idx == body_idx) continue;
                    
                    float other_x = shared_pos[2*j];
                    float other_y = shared_pos[2*j+1];
                    float other_mass = shared_mass[j];
                    
                    float dx = other_x - my_x;
                    float dy = other_y - my_y;
                    float dist_sq = dx*dx + dy*dy + SOFTENING;
                    float inv_dist = rsqrtf(dist_sq);
                    float inv_dist3 = inv_dist * inv_dist * inv_dist;
                    
                    float F = G * my_mass * other_mass * inv_dist3;
                    ax += F * dx;
                    ay += F * dy;
                }
                __syncthreads();
            }
            
            velocities[2*idx] += ax * dt;
            velocities[2*idx+1] += ay * dt;
            positions[2*idx] += velocities[2*idx] * dt;
            positions[2*idx+1] += velocities[2*idx+1] * dt;
        }
        """
        
        # Compile CUDA code with optimizations
        try:
            self.mod = SourceModule(cuda_code, options=['-O3', '-use_fast_math', '-arch=sm_75'])
        except pycuda.compiler.CompileError as e:
            print(f"Compilation failed: {e}")
            raise
        
        # Allocate device memory (pinned memory for faster transfers)
        self.d_positions = cuda.mem_alloc(self.positions_flat.nbytes)
        self.d_velocities = cuda.mem_alloc(self.velocities_flat.nbytes)
        self.d_masses = cuda.mem_alloc(self.masses.nbytes)
        
        # Copy data to device
        cuda.memcpy_htod(self.d_positions, self.positions_flat)
        cuda.memcpy_htod(self.d_velocities, self.velocities_flat)
        cuda.memcpy_htod(self.d_masses, self.masses)
        
        # Get the kernel function
        self.nbody_update = self.mod.get_function("nbody_update")
    
    def update(self) -> None:
        """Update all bodies using CUDA computation."""
        start_time = time.perf_counter()
        
        # Configure grid and block for optimal performance
        block_size = min(256, self.n_bodies)
        grid_size = (self.n_bodies + block_size - 1) // block_size
        
        # Execute kernel
        self.nbody_update(self.d_positions, self.d_velocities, self.d_masses,
                         np.int32(self.n_bodies), np.float32(self.G), np.float32(self.dt),
                         block=(block_size, 1, 1), grid=(grid_size, 1))
        
        # Copy results back to host (only positions for display)
        cuda.memcpy_dtoh(self.positions_flat, self.d_positions)
        
        # Update body objects (only positions for display)
        for i, body in enumerate(self.bodies):
            body.x = self.positions_flat[2*i]
            body.y = self.positions_flat[2*i+1]
        
        # Record performance (using circular buffer)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.performance_stats[self.stat_index] = elapsed_ms
        self.stat_index = (self.stat_index + 1) % len(self.performance_stats)
    
    def get_avg_time(self) -> float:
        """Get average computation time."""
        non_zero = self.performance_stats[self.performance_stats > 0]
        return np.mean(non_zero) if len(non_zero) > 0 else 0
    
    def reset(self) -> None:
        """Reset the simulation with new random bodies."""
        self.bodies = self._create_bodies()
        self.performance_stats.fill(0)
        self.stat_index = 0
        
        # Reinitialize arrays with new body data
        for i, body in enumerate(self.bodies):
            self.positions[i] = [body.x, body.y]
            self.velocities[i] = [body.vx, body.vy]
            self.masses[i] = body.mass
        
        # Update flattened arrays
        self.positions_flat = self.positions.flatten()
        self.velocities_flat = self.velocities.flatten()
        
        # Copy new data to device
        cuda.memcpy_htod(self.d_positions, self.positions_flat)
        cuda.memcpy_htod(self.d_velocities, self.velocities_flat)
        cuda.memcpy_htod(self.d_masses, self.masses)