import random
from dataclasses import dataclass
from typing import Tuple

# Constants
WIDTH, HEIGHT = 1000, 800
G = 6.674e-1   # Gravitational constant (scaled for visualization)
DT = 0.1       # Time step
N_BODIES = 10  # Number of bodies

@dataclass
class Body:
    """Represents a celestial body in the N-Body simulation."""
    x: float
    y: float
    mass: float
    radius: int
    color: Tuple[int, int, int]
    vx: float = 0.0
    vy: float = 0.0
    
    @classmethod
    def create_random(cls, width: int, height: int, min_mass: int = 5, max_mass: int = 20) -> 'Body':
        """Create a random body within the given boundaries."""
        return cls(
            x=random.randint(100, width-100),
            y=random.randint(100, height-100),
            mass=random.randint(min_mass, max_mass),
            radius=4,
            color=(
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255)
            ),
            vx=random.uniform(-1, 1),
            vy=random.uniform(-1, 1)
        )