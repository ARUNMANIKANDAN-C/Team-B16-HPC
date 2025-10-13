import numpy as np
from typing import List, Optional
from body import Body
import time
from quadtree_node import QuadTreeNode
from simulation_cpu import SimulationCPU


class SimulationCPUBarnesHut(SimulationCPU):
    """CPU-based N-Body simulation using Barnes-Hut algorithm (inherits from SimulationCPU)."""

    def __init__(self, width: int, height: int, G: float, dt: float, n_bodies: int, theta: float = 0.5):
        # Initialize parent class
        super().__init__(width, height, G, dt, n_bodies)
        
        # Barnes-Hut specific parameters
        self.theta = theta  # Opening angle parameter

    def _build_tree(self) -> QuadTreeNode:
        """Build Barnes-Hut quadtree from current bodies."""
        # Create root node covering entire simulation area
        root = QuadTreeNode(self.width / 2, self.height / 2, max(self.width, self.height))
        
        # Insert all bodies into the tree
        for body in self.bodies:
            root.insert(body)
        
        return root

    def update(self) -> None:
        """Update all bodies using Barnes-Hut algorithm (overrides parent method)."""
        start_time = time.perf_counter()

        # Build Barnes-Hut tree
        root = self._build_tree()

        # Calculate forces and update each body
        for body in self.bodies:
            # Calculate net force using Barnes-Hut approximation
            fx, fy = root.calculate_force(body, self.theta)
            
            # Apply gravitational constant
            fx *= self.G
            fy *= self.G
            
            # Update velocity (F = ma, so a = F/m)
            body.vx += (fx / body.mass) * self.dt
            body.vy += (fy / body.mass) * self.dt
            
            # Update position
            body.x += body.vx * self.dt
            body.y += body.vy * self.dt
            
            # Simple boundary handling
            self._handle_boundaries(body)

        # Record performance (using parent class method)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.performance_stats[self.stat_index] = elapsed_ms
        self.stat_index = (self.stat_index + 1) % len(self.performance_stats)

    def _handle_boundaries(self, body: Body) -> None:
        """Handle boundary conditions (simple bounce)."""
        bounce_damping = 0.9
        
        if body.x <= 0 or body.x >= self.width:
            body.vx *= -bounce_damping
            body.x = np.clip(body.x, 0, self.width)
        if body.y <= 0 or body.y >= self.height:
            body.vy *= -bounce_damping
            body.y = np.clip(body.y, 0, self.height)

    def set_theta(self, theta: float) -> None:
        """Set the Barnes-Hut opening angle parameter."""
        self.theta = max(0.1, min(2.0, theta))  # Clamp to reasonable values

    def get_algorithm_info(self) -> str:
        """Get information about the algorithm being used."""
        return f"Barnes-Hut Algorithm (θ={self.theta}) - O(N log N) complexity"