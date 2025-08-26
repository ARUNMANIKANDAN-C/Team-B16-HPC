import pygame
from typing import List, Dict
from body import Body, WIDTH, HEIGHT  # Import Body and constants from Part 1

class Visualization:
    """Handles Pygame visualization and event handling for the N-Body simulation."""
    
    def __init__(self):
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("N-Body Simulation (CUDA)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 16)
        
        # Pre-render static text
        self.bodies_text = self.font.render(f"Bodies: {10}", True, (255, 255, 255))  # N_BODIES hardcoded for now
        self.fps_prefix = self.font.render("FPS: ", True, (255, 255, 255))
        self.ms_prefix = self.font.render("Avg: ", True, (255, 255, 255))
        self.ms_suffix = self.font.render(" ms", True, (255, 255, 255))
        
        # Button definitions
        self.buttons = [
            {"rect": pygame.Rect(10, 10, 120, 30), "text": "Reset (R)", "key": pygame.K_r},
            {"rect": pygame.Rect(10, 50, 120, 30), "text": "Quit (ESC)", "key": pygame.K_ESCAPE}
        ]
        
        # Pre-render button text
        for button in self.buttons:
            button["text_surface"] = self.font.render(button["text"], True, (255, 255, 255))
            button["text_rect"] = button["text_surface"].get_rect(center=button["rect"].center)
    
    def handle_events(self) -> tuple[bool, bool]:
        """Handle Pygame events and return (running, reset_needed)."""
        running = True
        reset_needed = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    reset_needed = True
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = event.pos
                for button in self.buttons:
                    if button["rect"].collidepoint(mouse_pos):
                        if button["key"] == pygame.K_r:
                            reset_needed = True
                        elif button["key"] == pygame.K_ESCAPE:
                            running = False
        
        return running, reset_needed
    
    def render(self, bodies: List[Body], avg_time: float, fps: float) -> None:
        """Render the simulation state."""
        self.screen.fill((0, 0, 0))
        
        # Draw bodies efficiently
        for body in bodies:
            if -100 <= body.x <= WIDTH + 100 and -100 <= body.y <= HEIGHT + 100:
                pygame.draw.circle(self.screen, body.color, (int(body.x), int(body.y)), body.radius)
        
        # Draw buttons
        for button in self.buttons:
            pygame.draw.rect(self.screen, (50, 50, 50), button["rect"])
            pygame.draw.rect(self.screen, (100, 100, 100), button["rect"], 2)
            self.screen.blit(button["text_surface"], button["text_rect"])
        
        # Draw stats
        self.screen.blit(self.bodies_text, (10, 100))
        self.screen.blit(self.ms_prefix, (10, 125))
        self.screen.blit(self.fps_prefix, (10, 150))
        
        avg_text = self.font.render(f"{avg_time:.1f}", True, (255, 255, 255))
        fps_text = self.font.render(f"{fps:.1f}", True, (255, 255, 255))
        
        self.screen.blit(avg_text, (50, 125))
        self.screen.blit(self.ms_suffix, (50 + avg_text.get_width(), 125))
        self.screen.blit(fps_text, (50, 150))
        
        pygame.display.flip()
    
    def get_fps(self) -> float:
        """Return the current FPS."""
        return self.clock.get_fps()
    
    def tick(self) -> None:
        """Advance the clock without FPS limit."""
        self.clock.tick(0)
    
    def quit(self) -> None:
        """Clean up Pygame resources."""
        pygame.quit()