import cv2
import numpy as np
import time
import os
from typing import Tuple, List, Optional, Union
import colorsys
import imageio
from PIL import Image, ImageDraw, ImageFont
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class FluidColoredOpenCVPixelGenerator:
    """
    Enhanced pixel generator with fluid mechanics simulation for morphing between images.
    Uses particle-based fluid dynamics with viscosity, pressure, and flow fields.
    """
    
    def __init__(self, width: int = 120, height: int = 40):
        """
        Initialize the fluid pixel generator with particle system.
        
        Args:
            width: Output pixel grid width.
            height: Output pixel grid height.
        """
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive integers.")
        
        self.width = width
        self.height = height
        
        # Grid to store pixel colors
        self.color_grid = [[(255, 255, 255) for _ in range(width)] for _ in range(height)]
        
        # Fluid simulation parameters
        self.particles = []
        self.velocity_field = np.zeros((height, width, 2))
        self.pressure_field = np.zeros((height, width))
        self.density_field = np.zeros((height, width))
        
        # Fluid properties
        self.viscosity = 0.1
        self.particle_radius = 2.0
        self.rest_density = 1.0
        self.pressure_multiplier = 0.5
        self.damping = 0.99
        
    class FluidParticle:
        """Represents a single fluid particle with color and physics properties."""
        
        def __init__(self, x: float, y: float, color: Tuple[int, int, int], mass: float = 1.0):
            self.position = np.array([x, y], dtype=np.float32)
            self.velocity = np.array([0.0, 0.0], dtype=np.float32)
            self.force = np.array([0.0, 0.0], dtype=np.float32)
            self.color = color
            self.mass = mass
            self.density = 1.0
            self.pressure = 0.0
            
    def get_proper_dimensions(self, original_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate proper dimensions for resizing while maintaining aspect ratio."""
        orig_h, orig_w = original_shape
        
        if orig_h <= 0 or orig_w <= 0:
            return min(10, self.height), min(20, self.width)
        
        image_aspect = orig_w / orig_h
        target_aspect = self.width / self.height
        
        if image_aspect > target_aspect:
            target_w = self.width
            target_h = max(1, int(self.width / image_aspect))
            target_h = min(target_h, self.height)
        else:
            target_h = self.height
            target_w = max(1, int(self.height * image_aspect))
            target_w = min(target_w, self.width)
            
        target_h = max(1, min(target_h, self.height))
        target_w = max(1, min(target_w, self.width))
            
        return target_h, target_w
        
    def load_and_resize_image(self, image_path: str) -> np.ndarray:
        """Load image and resize with proper aspect ratio handling."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        target_h, target_w = self.get_proper_dimensions(img.shape[:2])
        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        return resized
        
    def create_velocity_field(self, progress: float) -> np.ndarray:
        """
        Create a fluid velocity field that guides particles from image1 to image2.
        Uses fluid dynamics principles including curl and divergence constraints.
        """
        y_coords, x_coords = np.mgrid[0:self.height, 0:self.width]
        
        # Base flow field with vorticity and streams
        base_flow_x = np.sin(2 * np.pi * x_coords / self.width) * np.cos(2 * np.pi * y_coords / self.height)
        base_flow_y = -np.cos(2 * np.pi * x_coords / self.width) * np.sin(2 * np.pi * y_coords / self.height)
        
        # Add turbulence and swirls for fluid-like motion
        turbulence_x = 0.3 * np.sin(4 * np.pi * x_coords / self.width + progress * 2 * np.pi)
        turbulence_y = 0.3 * np.cos(4 * np.pi * y_coords / self.height + progress * 2 * np.pi)
        
        # Progressive flow that moves particles toward target positions
        progress_flow_x = (progress - 0.5) * 2.0 * np.ones_like(x_coords)
        progress_flow_y = np.sin(progress * np.pi) * np.ones_like(y_coords)
        
        # Combine flows with fluid conservation
        velocity_x = (base_flow_x + turbulence_x + progress_flow_x) * (1 - progress) + progress_flow_x * progress
        velocity_y = (base_flow_y + turbulence_y + progress_flow_y) * (1 - progress) + progress_flow_y * progress
        
        # Apply divergence constraint to maintain incompressible flow
        velocity_x = self.apply_divergence_constraint(velocity_x, velocity_y)[0]
        velocity_y = self.apply_divergence_constraint(velocity_x, velocity_y)[1]
        
        return np.stack([velocity_x, velocity_y], axis=-1)
        
    def apply_divergence_constraint(self, vx: np.ndarray, vy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply divergence-free constraint to velocity field for incompressible flow."""
        # Calculate divergence
        div_vx = np.gradient(vx, axis=1)
        div_vy = np.gradient(vy, axis=0)
        divergence = div_vx + div_vy
        
        # Correct velocities to reduce divergence
        correction_factor = 0.5
        vx_corrected = vx - correction_factor * np.gradient(divergence, axis=1)
        vy_corrected = vy - correction_factor * np.gradient(divergence, axis=0)
        
        return vx_corrected, vy_corrected
        
    def initialize_particles_from_image(self, img: np.ndarray) -> List[FluidParticle]:
        """Create fluid particles from image pixels with proper spacing."""
        particles = []
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Sample particles with some spacing to avoid overcrowding
        particle_spacing = max(1, min(h, w) // 20)
        
        offset_y = (self.height - h) // 2
        offset_x = (self.width - w) // 2
        
        for y in range(0, h, particle_spacing):
            for x in range(0, w, particle_spacing):
                target_y = y + offset_y
                target_x = x + offset_x
                
                if 0 <= target_y < self.height and 0 <= target_x < self.width:
                    r, g, b = img_rgb[y, x]
                    color = (int(r), int(g), int(b))
                    
                    # Add some randomness to particle positions for natural fluid look
                    px = target_x + np.random.uniform(-0.5, 0.5)
                    py = target_y + np.random.uniform(-0.5, 0.5)
                    
                    particle = self.FluidParticle(px, py, color)
                    particles.append(particle)
                    
        return particles
        
    def calculate_fluid_forces(self, particles: List[FluidParticle]) -> None:
        """Calculate fluid dynamics forces: pressure, viscosity, and external forces."""
        # Reset forces
        for particle in particles:
            particle.force = np.array([0.0, 0.0])
            
        # Calculate density and pressure for each particle
        for i, particle in enumerate(particles):
            density = 0.0
            for j, other in enumerate(particles):
                if i != j:
                    distance = np.linalg.norm(particle.position - other.position)
                    if distance < self.particle_radius:
                        # SPH kernel function (simplified)
                        h = self.particle_radius
                        kernel_value = max(0, (h - distance) ** 3) / (h ** 3)
                        density += other.mass * kernel_value
                        
            particle.density = max(density, self.rest_density * 0.1)
            particle.pressure = self.pressure_multiplier * (particle.density - self.rest_density)
            
        # Calculate pressure and viscosity forces
        for i, particle in enumerate(particles):
            pressure_force = np.array([0.0, 0.0])
            viscosity_force = np.array([0.0, 0.0])
            
            for j, other in enumerate(particles):
                if i != j:
                    distance_vec = particle.position - other.position
                    distance = np.linalg.norm(distance_vec)
                    
                    if distance < self.particle_radius and distance > 0:
                        direction = distance_vec / distance
                        h = self.particle_radius
                        
                        # Pressure force (repulsive)
                        pressure_gradient = (particle.pressure + other.pressure) / (2 * other.density)
                        kernel_gradient = -3 * (h - distance) ** 2 / (h ** 3)
                        pressure_force += -other.mass * pressure_gradient * kernel_gradient * direction
                        
                        # Viscosity force (smoothing)
                        velocity_diff = other.velocity - particle.velocity
                        viscosity_laplacian = 6 * (h - distance) / (h ** 3)
                        viscosity_force += self.viscosity * other.mass * velocity_diff * viscosity_laplacian / other.density
                        
            particle.force += pressure_force + viscosity_force
            
    def update_particle_physics(self, particles: List[FluidParticle], dt: float, velocity_field: np.ndarray) -> None:
        """Update particle positions and velocities using fluid dynamics."""
        for particle in particles:
            # Get velocity field influence at particle position
            x, y = int(np.clip(particle.position[0], 0, self.width - 1)), int(np.clip(particle.position[1], 0, self.height - 1))
            field_velocity = velocity_field[y, x]
            
            # Apply external velocity field as additional force
            external_force = field_velocity * 0.5
            
            # Total acceleration from all forces
            total_force = particle.force + external_force
            acceleration = total_force / particle.mass
            
            # Verlet integration for better stability
            new_velocity = particle.velocity + acceleration * dt
            new_velocity *= self.damping  # Apply damping
            
            new_position = particle.position + new_velocity * dt
            
            # Boundary conditions with reflection
            if new_position[0] < 0 or new_position[0] >= self.width:
                new_velocity[0] *= -0.8
                new_position[0] = np.clip(new_position[0], 0, self.width - 1)
                
            if new_position[1] < 0 or new_position[1] >= self.height:
                new_velocity[1] *= -0.8
                new_position[1] = np.clip(new_position[1], 0, self.height - 1)
                
            particle.velocity = new_velocity
            particle.position = new_position
            
    def render_particles_to_grid(self, particles: List[FluidParticle]) -> None:
        """Render fluid particles to the color grid with smooth blending."""
        # Clear grid
        self.color_grid = [[(0, 0, 0) for _ in range(self.width)] for _ in range(self.height)]
        
        # Create influence maps for smooth rendering
        color_accumulator = np.zeros((self.height, self.width, 3))
        weight_accumulator = np.zeros((self.height, self.width))
        
        for particle in particles:
            px, py = particle.position
            color = np.array(particle.color)
            
            # Render particle with smooth falloff
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    gx, gy = int(px) + dx, int(py) + dy
                    
                    if 0 <= gx < self.width and 0 <= gy < self.height:
                        distance = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
                        if distance < 2.5:
                            weight = max(0, 1 - distance / 2.5)
                            color_accumulator[gy, gx] += color * weight
                            weight_accumulator[gy, gx] += weight
                            
        # Normalize colors
        for y in range(self.height):
            for x in range(self.width):
                if weight_accumulator[y, x] > 0:
                    final_color = color_accumulator[y, x] / weight_accumulator[y, x]
                    self.color_grid[y][x] = tuple(np.clip(final_color, 0, 255).astype(int))
                else:
                    self.color_grid[y][x] = (0, 0, 0)
                    
    def fluid_morph_images(self, img1_path: str, img2_path: str, progress: float, 
                          simulation_steps: int = 5) -> None:
        """
        Perform fluid-based morphing between two images using particle simulation.
        
        Args:
            img1_path: Path to source image
            img2_path: Path to target image
            progress: Morphing progress (0.0 to 1.0)
            simulation_steps: Number of physics simulation steps per frame
        """
        img1 = self.load_and_resize_image(img1_path)
        img2 = self.load_and_resize_image(img2_path)
        
        # Create velocity field for this progress step
        velocity_field = self.create_velocity_field(progress)
        
        # Initialize particles from source image at progress=0
        if not hasattr(self, 'fluid_particles') or progress == 0:
            self.fluid_particles = self.initialize_particles_from_image(img1)
            
        # Get target colors from img2 for color interpolation
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        target_h, target_w = self.get_proper_dimensions(img2.shape[:2])
        img2_resized = cv2.resize(img2_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        # Update particle colors based on progress
        for particle in self.fluid_particles:
            if hasattr(particle, 'original_color'):
                orig_color = particle.original_color
            else:
                particle.original_color = particle.color
                orig_color = particle.color
                
            # Sample target color
            x, y = int(np.clip(particle.position[0], 0, target_w - 1)), int(np.clip(particle.position[1], 0, target_h - 1))
            if y < target_h and x < target_w:
                target_color = img2_resized[y, x]
                
                # Interpolate colors
                current_color = (
                    int(orig_color[0] * (1 - progress) + target_color[0] * progress),
                    int(orig_color[1] * (1 - progress) + target_color[1] * progress),
                    int(orig_color[2] * (1 - progress) + target_color[2] * progress)
                )
                particle.color = current_color
                
        # Run fluid simulation steps
        dt = 0.016 / simulation_steps  # 60 FPS divided by simulation steps
        
        for step in range(simulation_steps):
            self.calculate_fluid_forces(self.fluid_particles)
            self.update_particle_physics(self.fluid_particles, dt, velocity_field)
            
        # Render particles to grid
        self.render_particles_to_grid(self.fluid_particles)
        
    def render_to_image(self, bg_color=(0, 0, 0), pixel_size=5) -> Image.Image:
        """Render the color grid to a PIL image."""
        img_w = self.width * pixel_size
        img_h = self.height * pixel_size
        
        img = Image.new("RGB", (img_w, img_h), bg_color)
        draw = ImageDraw.Draw(img)
        
        for y in range(self.height):
            for x in range(self.width):
                r, g, b = self.color_grid[y][x]
                draw.rectangle([x * pixel_size, y * pixel_size,
                                (x + 1) * pixel_size, (y + 1) * pixel_size],
                                fill=(r, g, b))
                                
        return img
        
    def save_fluid_morph_gif(self, img1_path: str, img2_path: str, steps: int = 30,
                           delay: float = 0.1, output_path: str = "fluid_morph.gif",
                           pixel_size: int = 5, simulation_steps: int = 3) -> None:
        """
        Save a fluid-based morph animation as a GIF.
        
        Args:
            img1_path, img2_path: Input image paths
            steps: Number of animation frames
            delay: Delay per frame (seconds)
            output_path: Path to save GIF
            pixel_size: Size of each pixel in output
            simulation_steps: Physics simulation steps per frame
        """
        frames = []
        
        print("Generating fluid morph animation...")
        for i in range(steps):
            progress = i / (steps - 1)
            print(f"Processing frame {i+1}/{steps} (progress: {progress:.2f})")
            
            self.fluid_morph_images(img1_path, img2_path, progress, simulation_steps)
            frame_img = self.render_to_image(pixel_size=pixel_size)
            frames.append(frame_img)
        
        if frames:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=int(delay * 1000),
                loop=0
            )
            print(f"Saved fluid morph GIF to {output_path}")
            
    def save_as_html_grid(self, filename: str) -> None:
        """Save the pixel grid as an HTML file."""
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ background-color: black; margin: 0; display: flex; justify-content: center; align-items: center; height: 100vh; }}
        .pixel-grid-container {{ line-height: 0; white-space: nowrap; border: 1px solid white; }}
        .pixel {{
            display: inline-block;
            width: 5px;
            height: 5px;
            margin: 0;
            padding: 0;
            border: none;
        }}
    </style>
</head>
<body>
    <div class="pixel-grid-container">"""
    
        for y in range(len(self.color_grid)):
            for x in range(len(self.color_grid[y])):
                r, g, b = self.color_grid[y][x]
                html_content += f'<span class="pixel" style="background-color: rgb({r},{g},{b})"></span>'
            html_content += '\n'
    
        html_content += """    </div>
</body>
</html>"""
    
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Saved fluid pixel art HTML grid to {filename}")

# Example usage
if __name__ == "__main__":
    try:
        # Create dummy images if they don't exist
        img1_path = "image1.jpg"
        img2_path = "image2.jpg"
        
        if not os.path.exists(img1_path):
            img1 = Image.new('RGB', (200, 200), 'blue')
            draw = ImageDraw.Draw(img1)
            draw.rectangle([50, 50, 150, 150], fill='yellow')
            img1.save(img1_path)
            
        if not os.path.exists(img2_path):
            img2 = Image.new('RGB', (200, 200), 'red')
            draw = ImageDraw.Draw(img2)
            draw.ellipse([50, 50, 150, 150], fill='green')
            img2.save(img2_path)
            
        print("Fluid-Enhanced OpenCV Art Generator")
        print("=" * 50)
        
        # Create fluid morph animation
        print("Generating fluid-based morphing GIF...")
        fluid_gen = FluidColoredOpenCVPixelGenerator(width=100, height=60)
        fluid_gen.save_fluid_morph_gif(
            img1_path, img2_path, 
            steps=100, 
            delay=0.15,
            output_path="fluid_pixel_morph.gif", 
            pixel_size=1,
            simulation_steps=15
        )
        
    except Exception as e:
        print(f"An error occurred: {e}")
