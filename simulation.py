import numpy as np
import moderngl
from pathlib import Path
from typing import Tuple

class GalaxySimulation:
    def __init__(self, num_stars=30000):
        """Initialize GPU-accelerated galaxy simulation."""
        # Simulation parameters
        self.num_stars = num_stars
        self.gravity = 2.0
        self.time_step = 0.0005
        self.black_hole_force = 20.0
        self.interaction_rate = 0.3  # Reduce for better performance
        self.initial_radius = 200.0  # Larger initial radius
        self.initial_height = 20.0   # Thicker disk
        self.middle_velocity = 1.0    # Initial orbital velocities
        self.outer_velocity = 3.0     # Outer rim velocity

        # Create ModernGL standalone context (no window needed)
        self.ctx = moderngl.create_standalone_context()
        
        # Calculate dimensions for compute textures
        self.work_group_size = 16
        self.size = int(np.ceil(np.sqrt(num_stars) / self.work_group_size)) * self.work_group_size
        self.actual_num_stars = self.size * self.size
        self.num_work_groups = self.size // self.work_group_size

        # Create compute shaders directory
        shader_dir = Path(__file__).parent / 'shaders'
        shader_dir.mkdir(exist_ok=True)

        # Create position and velocity textures
        self.position_texture = self.ctx.texture((self.size, self.size), 4, dtype='f4')
        self.velocity_texture = self.ctx.texture((self.size, self.size), 4, dtype='f4')
        
        # Create buffer bindings for compute shaders
        self.position_buffer = self.ctx.buffer(reserve=self.actual_num_stars * 16)  # 4 floats * 4 bytes
        self.velocity_buffer = self.ctx.buffer(reserve=self.actual_num_stars * 16)
        
        # Load compute shaders
        self.velocity_program = self._create_compute_shader('computeShaderVelocity.glsl')
        self.position_program = self._create_compute_shader('computeShaderPosition.glsl')
        
        # Initialize particle positions and velocities
        self._initialize_particles()

    def _create_texture(self, size: Tuple[int, int], components: int) -> moderngl.Texture:
        """Create a floating point texture for compute shader."""
        return self.ctx.texture(size, components, dtype='f4')

    def _create_compute_shader(self, shader_name: str) -> moderngl.ComputeShader:
        """Load and create a compute shader."""
        shader_path = Path(__file__).parent / 'shaders' / shader_name
        if not shader_path.exists():
            # Copy shader from reference implementation
            ref_shader_path = Path(__file__).parent.parent / 'galaxy_sim' / 'src' / 'shaders' / shader_name
            if ref_shader_path.exists():
                shader_text = ref_shader_path.read_text()
                # Remove export default and template literals
                shader_text = shader_text.replace('export default `', '').rstrip('`')
                shader_path.write_text(shader_text)
        
        with open(shader_path) as f:
            return self.ctx.compute_shader(f.read())

    def _initialize_particles(self):
        """Initialize particle positions and velocities in a galaxy shape."""
        positions = np.zeros((self.actual_num_stars, 4), dtype=np.float32)
        velocities = np.zeros((self.actual_num_stars, 4), dtype=np.float32)

        for i in range(self.num_stars):
            # Generate random polar coordinates
            radius = np.random.uniform(0, self.initial_radius)
            angle = np.random.uniform(0, 2 * np.pi)
            height = np.random.uniform(-self.initial_height, self.initial_height)
            
            # Convert to Cartesian coordinates
            x = radius * np.cos(angle)
            y = height
            z = radius * np.sin(angle)
            
            # Calculate initial velocity for circular orbit
            velocity_factor = self.middle_velocity
            if radius > self.initial_radius * 0.8:
                velocity_factor = self.outer_velocity
                
            vx = -velocity_factor * np.sin(angle)
            vz = velocity_factor * np.cos(angle)
            vy = 0
            
            positions[i] = [x, y, z, 1.0]  # w=1.0 for regular matter
            velocities[i] = [vx, vy, vz, 0.0]  # w=0.0 for initial acceleration

        # Upload to textures
        self.position_texture.write(positions.tobytes())
        self.velocity_texture.write(velocities.tobytes())

    def step(self):
        """Perform one simulation step."""
        # Bind textures to compute shaders
        self.position_texture.bind_to_image(0)
        self.velocity_texture.bind_to_image(1)
        
        # Update velocities
        self.velocity_program['deltaTime'] = self.time_step
        self.velocity_program['gravity'] = self.gravity
        self.velocity_program['interactionRate'] = self.interaction_rate
        self.velocity_program['blackHoleForce'] = self.black_hole_force
        self.velocity_program['maxAcceleration'] = 1000.0  # For color calculation
        self.velocity_program.run(group_x=self.num_work_groups, group_y=self.num_work_groups)
        
        # Update positions
        self.position_program['deltaTime'] = self.time_step
        self.position_program.run(group_x=self.num_work_groups, group_y=self.num_work_groups)
        
        # Ensure compute is finished before proceeding
        self.ctx.finish()
    
    def get_particle_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current particle positions and velocities."""
        # Make sure we're reading from the latest data
        self.ctx.finish()
        
        # Read from textures
        positions = np.frombuffer(
            self.position_texture.read(), 
            dtype=np.float32
        ).reshape(self.actual_num_stars, 4)
        
        velocities = np.frombuffer(
            self.velocity_texture.read(), 
            dtype=np.float32
        ).reshape(self.actual_num_stars, 4)
        
        # Return only the active particles
        return positions[:self.num_stars], velocities[:self.num_stars]    
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'ctx'):
            self.ctx.release()
