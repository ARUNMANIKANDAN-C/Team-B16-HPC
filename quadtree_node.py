import numpy as np
from typing import Optional
from body import Body
class QuadTreeNode:
    """Node for Barnes-Hut quadtree."""
    
    def __init__(self, center_x: float, center_y: float, size: float):
        self.center_x = center_x
        self.center_y = center_y
        self.size = size  # Width/height of this quadrant
        
        # Body data (for leaf nodes)
        self.body: Optional[Body] = None
        self.total_mass = 0.0
        self.center_of_mass_x = 0.0
        self.center_of_mass_y = 0.0
        
        # Children (NW, NE, SW, SE)
        self.nw: Optional[QuadTreeNode] = None
        self.ne: Optional[QuadTreeNode] = None
        self.sw: Optional[QuadTreeNode] = None
        self.se: Optional[QuadTreeNode] = None
        self.is_leaf = True
    
    def insert(self, body: Body) -> None:
        """Insert a body into the quadtree."""
        if self.total_mass == 0:
            # Empty node, store body here
            self.body = body
            self.total_mass = body.mass
            self.center_of_mass_x = body.x
            self.center_of_mass_y = body.y
            return
        
        if self.is_leaf:
            # Leaf node with existing body, need to split
            self._split()
            # Re-insert the existing body
            if self.body:
                self._insert_into_child(self.body)
                self.body = None
        
        # Insert new body into appropriate child
        self._insert_into_child(body)
        
        # Update center of mass
        self._update_center_of_mass(body)
    
    def _split(self) -> None:
        """Split the current node into four children."""
        half_size = self.size / 2
        quarter_size = self.size / 4
        
        # Create four children
        self.nw = QuadTreeNode(
            self.center_x - quarter_size, 
            self.center_y - quarter_size, 
            half_size
        )
        self.ne = QuadTreeNode(
            self.center_x + quarter_size, 
            self.center_y - quarter_size, 
            half_size
        )
        self.sw = QuadTreeNode(
            self.center_x - quarter_size, 
            self.center_y + quarter_size, 
            half_size
        )
        self.se = QuadTreeNode(
            self.center_x + quarter_size, 
            self.center_y + quarter_size, 
            half_size
        )
        
        self.is_leaf = False
    
    def _insert_into_child(self, body: Body) -> None:
        """Insert a body into the appropriate child node."""
        if body.x <= self.center_x:
            if body.y <= self.center_y:
                self.nw.insert(body)
            else:
                self.sw.insert(body)
        else:
            if body.y <= self.center_y:
                self.ne.insert(body)
            else:
                self.se.insert(body)
    
    def _update_center_of_mass(self, body: Body) -> None:
        """Update the center of mass after inserting a new body."""
        total_mass = self.total_mass + body.mass
        self.center_of_mass_x = (
            self.center_of_mass_x * self.total_mass + body.x * body.mass
        ) / total_mass
        self.center_of_mass_y = (
            self.center_of_mass_y * self.total_mass + body.y * body.mass
        ) / total_mass
        self.total_mass = total_mass
    
    def calculate_force(self, body: Body, theta: float = 0.5) -> tuple[float, float]:
        """Calculate force on a body using Barnes-Hut approximation."""
        if self.total_mass == 0 or body is self.body:
            return 0.0, 0.0
        
        dx = self.center_of_mass_x - body.x
        dy = self.center_of_mass_y - body.y
        dist_sq = dx**2 + dy**2
        
        # Avoid division by zero
        if dist_sq < 1e-10:
            return 0.0, 0.0
        
        distance = np.sqrt(dist_sq)
        
        # If node is sufficiently far or is a leaf, use approximation
        if self.size / distance < theta or self.is_leaf:
            force_mag = body.mass * self.total_mass / dist_sq
            fx = force_mag * dx / distance
            fy = force_mag * dy / distance
            return fx, fy
        
        # Otherwise, traverse children
        fx_total, fy_total = 0.0, 0.0
        for child in [self.nw, self.ne, self.sw, self.se]:
            if child and child.total_mass > 0:
                fx, fy = child.calculate_force(body, theta)
                fx_total += fx
                fy_total += fy
        
        return fx_total, fy_total
