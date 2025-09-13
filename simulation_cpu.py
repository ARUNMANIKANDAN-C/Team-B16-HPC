import numpy as np
from typing import List
from body import Body
import time

class SimulationCPU:
    """CPU-based N-Body simulation using only Python and Numpy."""

    def __init__(self, width: int, height: int, G: float, dt: float, n_bodies: int):
        self.width = width
        self.height = height
        self.G = G
        self.dt = dt
        self.n_bodies = n_bodies
        self.bodies = self._create_bodies()
        self.performance_stats = np.zeros(100, dtype=np.float32)
        self.stat_index = 0

    def _create_bodies(self) -> List[Body]:
        """Create initial set of bodies."""
        return [Body.create_random(self.width, self.height) for _ in range(self.n_bodies)]

    def update(self) -> None:
        """Update all bodies using CPU computation."""
        start_time = time.perf_counter()

        # Extract positions, velocities, masses
        pos = np.array([[b.x, b.y] for b in self.bodies], dtype=np.float32)
        vel = np.array([[b.vx, b.vy] for b in self.bodies], dtype=np.float32)
        mass = np.array([b.mass for b in self.bodies], dtype=np.float32)

        # Initialize acceleration array
        acc = np.zeros_like(pos, dtype=np.float32)
        softening = 1e-3

        # Compute pairwise forces
        for i in range(self.n_bodies):
            dx = pos[:, 0] - pos[i, 0]
            dy = pos[:, 1] - pos[i, 1]
            dist_sq = dx**2 + dy**2 + softening
            inv_dist3 = 1.0 / (dist_sq * np.sqrt(dist_sq))
            inv_dist3[i] = 0  # Skip self-interaction
            F = self.G * mass[i] * mass * inv_dist3
            acc[i, 0] = np.sum(F * dx)
            acc[i, 1] = np.sum(F * dy)

        # Update velocities and positions
        vel += acc * self.dt
        pos += vel * self.dt

        # Update body objects
        for i, b in enumerate(self.bodies):
            b.x = pos[i, 0]
            b.y = pos[i, 1]
            b.vx = vel[i, 0]
            b.vy = vel[i, 1]

        # Record performance
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.performance_stats[self.stat_index] = elapsed_ms
        self.stat_index = (self.stat_index + 1) % len(self.performance_stats)

    def get_avg_time(self) -> float:
        """Get average computation time (ms)."""
        non_zero = self.performance_stats[self.performance_stats > 0]
        return float(np.mean(non_zero)) if len(non_zero) > 0 else 0

    def reset(self) -> None:
        """Reset the simulation with new random bodies."""
        self.bodies = self._create_bodies()
        self.performance_stats.fill(0)
        self.stat_index = 0
