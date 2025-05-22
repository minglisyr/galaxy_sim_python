import numpy as np
import moderngl
from pathlib import Path
from typing import Tuple

class GalaxySimulation:    
    def __init__(self, num_stars=30000):
        """Initialize GPU-accelerated galaxy simulation."""
        # Simulation parameters (matching reference implementation)
        self.num_stars = num_stars
        self.gravity = 0.05         # Reduced gravity
        self.time_step = 0.0005     # Smaller timestep for smoother motion
        self.black_hole_force = 0.5  # Reduced black hole force
        self.interaction_rate = 1.0
        self.initial_radius = 20.0
        self.initial_height = 3.0
        self.middle_velocity = 1.0   # Reduced middle velocity
        self.outer_velocity = 5.0    # Significantly reduced outer velocity

        # Create ModernGL standalone context (no window needed)
        self.ctx = moderngl.create_standalone_context()
        
        # Calculate texture dimensions based on workgroup size
        self.WORKGROUP_SIZE = 16
        texture_size = int(np.ceil(np.sqrt(num_stars)))
        # Round up to nearest multiple of workgroup size
        self.texture_width = ((texture_size + self.WORKGROUP_SIZE - 1) 
                            // self.WORKGROUP_SIZE * self.WORKGROUP_SIZE)
        self.texture_height = self.texture_width
        
        # Calculate work group counts
        self.num_groups_x = (self.texture_width + self.WORKGROUP_SIZE - 1) // self.WORKGROUP_SIZE
        self.num_groups_y = (self.texture_height + self.WORKGROUP_SIZE - 1) // self.WORKGROUP_SIZE
    
        # Calculate total size needed for textures
        self.total_pixels = self.texture_width * self.texture_height
    
        # Create textures with correct format
        self.position_texture = self.ctx.texture(
            (self.texture_width, self.texture_height), 4, 
            dtype='f4'
        )
        self.velocity_texture = self.ctx.texture(
            (self.texture_width, self.texture_height), 4, 
            dtype='f4'
        )
        
        # Create buffer bindings for compute shaders
        self.position_buffer = self.ctx.buffer(reserve=self.num_stars * 16)  # 4 floats * 4 bytes
        self.velocity_buffer = self.ctx.buffer(reserve=self.num_stars * 16)
        
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
        # Create arrays sized to texture dimensions
        positions = np.zeros((self.total_pixels, 4), dtype=np.float32)
        velocities = np.zeros((self.total_pixels, 4), dtype=np.float32)

        # Create central black hole
        positions[0] = [0.0, 0.0, 0.0, 1.0]  # w=1.0 marks it as dark matter
        velocities[0] = [0.0, 0.0, 0.0, 0.0]

        # Create galaxy spiral arms
        for i in range(1, self.num_stars):
            # Generate random polar coordinates
            radius = np.random.uniform(0, self.initial_radius)
            angle = np.random.uniform(0, 2 * np.pi)
            height = np.random.uniform(-self.initial_height, self.initial_height)
            
            # Add some spiral structure with randomness
            spiral_factor = 2.0
            angle += spiral_factor * radius / self.initial_radius
            
            # Convert to Cartesian coordinates
            x = radius * np.cos(angle)
            y = height
            z = radius * np.sin(angle)
            
            # Calculate initial velocity for circular orbit
            velocity_factor = self.middle_velocity
            if radius > self.initial_radius * 0.8:
                velocity_factor = self.outer_velocity
                
            # Tangential velocity for circular orbit
            vx = -velocity_factor * np.sin(angle)
            vz = velocity_factor * np.cos(angle)
            vy = np.random.normal(0, 0.1)  # Small random vertical velocity
            positions[i] = [x, y, z, 0.0]  # w=0.0 for regular matter
            velocities[i] = [vx, vy, vz, 0.0]  # w component stores acceleration magnitude

        # Pad remaining texture pixels with zeros
        # (positions and velocities arrays are already zero-initialized)

        # Reshape arrays to match texture dimensions
        positions_2d = positions.reshape(self.texture_height, self.texture_width, 4)
        velocities_2d = velocities.reshape(self.texture_height, self.texture_width, 4)

        # Upload to textures
        self.position_texture.write(positions_2d.tobytes())
        self.velocity_texture.write(velocities_2d.tobytes())

    def step(self):
        """Perform one simulation step."""
        # Bind textures
        self.position_texture.bind_to_image(0, read=True, write=True)
        self.velocity_texture.bind_to_image(1, read=True, write=True)
        
        # Update uniforms if they change
        self.velocity_program['deltaTime'] = self.time_step
        self.velocity_program['gravity'] = self.gravity
        self.velocity_program['interactionRate'] = self.interaction_rate
        self.velocity_program['blackHoleForce'] = self.black_hole_force
        
        # Run velocity compute shader
        self.velocity_program.run(self.num_groups_x, self.num_groups_y, 1)
        
        # Ensure velocity computation is complete
        self.ctx.finish()
        
        # Update positions
        self.position_program['deltaTime'] = self.time_step
        self.position_program.run(self.num_groups_x, self.num_groups_y, 1)
        
        # Ensure all computations are finished before proceeding
        self.ctx.finish()
    
    def get_particle_data(self):
        """Read back particle data from GPU textures."""
        # Read texture data
        position_data = np.frombuffer(
            self.position_texture.read(), 
            dtype=np.float32
        ).reshape(self.texture_height, self.texture_width, 4)
        
        velocity_data = np.frombuffer(
            self.velocity_texture.read(), 
            dtype=np.float32
        ).reshape(self.texture_height, self.texture_width, 4)
        
        # Flatten to particle arrays
        positions = position_data.reshape(-1, 4)[:self.num_stars]
        velocities = velocity_data.reshape(-1, 4)[:self.num_stars]
        
        return positions, velocities

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'ctx'):
            self.ctx.release()
