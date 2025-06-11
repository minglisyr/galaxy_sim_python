import moderngl
import numpy as np
import glfw        
import sys
import pyrr
from typing import Tuple
from pathlib import Path

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

class GalaxyVisualization:
    def __init__(self, width=800, height=600):
        # Store window dimensions first
        self.width = width
        self.height = height

        # Initialize GLFW and create window
        if not glfw.init():
            sys.exit("Could not initialize GLFW")

        # Configure GLFW
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, 4)
        glfw.window_hint(glfw.VISIBLE, True)

        # Create window
        self.window = glfw.create_window(self.width, self.height, "Galaxy Simulation", None, None)
        if not self.window:
            glfw.terminate()
            sys.exit("Could not create window")

        # Set up OpenGL context
        glfw.make_context_current(self.window)
        self.ctx = moderngl.create_context()
        
        # Enable necessary OpenGL features
        self.ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self.ctx.point_size = 1.0

        # Create simulation first, before shader setup
        self.simulation = GalaxySimulation()

        # Initialize camera parameters
        self.camera_distance = 300.0  # Increased initial distance for better view
        self.camera_position = np.array([15.0, 100.0, 300.0], dtype=np.float32)
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.camera_rotation = np.array([0.3, 0.0, 0.0], dtype=np.float32)  # Slight tilt
        self.last_mouse = None

        # Set up projection matrix
        aspect_ratio = self.width / self.height
        self.proj = pyrr.matrix44.create_perspective_projection(
            fovy=60.0,
            aspect=aspect_ratio,
            near=0.1,
            far=1000.0
        )

        # Set up camera view matrix
        self.setup_camera()

        # Now create rendering programs after simulation is initialized
        self.setup_particle_rendering()

        # Update uniforms
        self.render_program['projection'].write(self.proj.astype('f4').tobytes())
        self.render_program['pointSize'].value = 1.0
        self.render_program['view'].write(self.view.astype('f4').tobytes())

        # Set up input callbacks
        glfw.set_scroll_callback(self.window, self.handle_scroll)

        # Pre-allocate arrays for particle updates
        self.colors = np.zeros((self.simulation.num_stars, 4), dtype=np.float32)
        self.vertex_data = np.zeros((self.simulation.num_stars, 7), dtype=np.float32)

    def setup_camera(self):
        # Use consistent values with __init__
        self.proj = pyrr.matrix44.create_perspective_projection(
            fovy=60.0,
            aspect=self.window_ratio,
            near=0.1,
            far=1000.0
        )
        self.update_view_matrix()

    def update_view_matrix(self):
        # Create view matrix
        rotation = pyrr.matrix44.create_from_eulers(self.camera_rotation)
        translation = pyrr.matrix44.create_from_translation([0, 0, -self.camera_distance])
        self.view = pyrr.matrix44.multiply(rotation, translation)
        # Update shader uniforms immediately
        if hasattr(self, 'render_program'):
            self.render_program['view'].write(self.view.astype('f4').tobytes())

    @property
    def window_ratio(self):
        width, height = glfw.get_window_size(self.window)
        return width / height

    def setup_particle_rendering(self):
        try:
            # Load shaders from files
            shader_dir = Path(__file__).parent / 'shaders'
            with open(shader_dir / 'vertex.glsl') as f:
                vertex_shader = f.read()
            with open(shader_dir / 'fragment.glsl') as f:
                fragment_shader = f.read()

            self.render_program = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader
            )
            print("\nShader compilation successful!")

            # Set up uniforms with correct names (matching shader)
            self.render_program['projection'].write(self.proj.astype('f4').tobytes())
            self.render_program['pointSize'].value = 1.0  
            self.render_program['brightness'].value = 1.0  
            
            # Debug print uniform values
            print("\nUniform values:")
            print(f"Projection matrix:\n{self.proj}")
            print(f"Initial view matrix:\n{self.view}")
            print(f"Point size: {self.render_program['pointSize'].value}")
            
            # Create vertex buffer for particles
            self.vertex_buffer = self.ctx.buffer(reserve=self.simulation.num_stars * 28)
            self.vao = self.ctx.vertex_array(
                self.render_program,
                [(self.vertex_buffer, '3f 4f', 'position', 'color')]
            )
            print(f"\nVAO created successfully with format: '3f 4f'")
            print(f"Buffer size: {self.simulation.num_stars * 28} bytes")

        except Exception as e:
            print(f"\nError in shader setup: {str(e)}")
            raise

    def update_particle_buffer(self):
        positions, velocities = self.simulation.get_particle_data()
        
        # Debug information
        if len(positions) > 0:
            print("\nDebug Rendering Info:")
            print(f"Number of particles: {len(positions)}")
            print(f"Position range: min={positions[:, :3].min()}, max={positions[:, :3].max()}")
            print(f"Velocities range: min={velocities[:, :3].min()}, max={velocities[:, :3].max()}")
            print(f"First few positions:")
            print(positions[:3, :3])

        # Calculate colors based on velocity
        speeds = np.linalg.norm(velocities[:, :3], axis=1)
        max_speed = np.max(speeds) if len(speeds) > 0 else 1.0
        colors = np.zeros((len(positions), 4), dtype=np.float32)
        
        # Calculate normalized distance from center for color
        distances = np.linalg.norm(positions[:, :3], axis=1)
        max_dist = np.max(distances) if len(distances) > 0 else 1.0
        dist_factor = distances / max_dist

        # Create a blue-white color gradient based on distance and speed
        colors[:, 0] = 0.5 + 0.5 * (speeds / max_speed)  # Red
        colors[:, 1] = 0.5 + 0.5 * (speeds / max_speed)  # Green
        colors[:, 2] = 0.8 + 0.2 * (1.0 - dist_factor)   # Blue (brighter at center)
        colors[:, 3] = 1.0  # Full alpha

        # Combine positions and colors
        vertex_data = np.hstack([positions[:, :3], colors])
        self.vertex_buffer.write(vertex_data.astype('f4').tobytes())

    def handle_scroll(self, window, xoffset, yoffset):
        # Smoother zoom with smaller steps
        zoom_factor = 0.95 if yoffset > 0 else 1.05
        self.camera_distance = max(50.0, min(
            self.camera_distance * zoom_factor,
            500.0  # Allow zooming out further
        ))
        self.update_view_matrix()
        # Should update shader uniforms here
        self.render_program['view'].write(self.view.astype('f4').tobytes())

    def render(self):
        # Clear screen and depth buffer
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)  # Black background
        
        # Get updated particle data and update buffer
        self.update_particle_buffer()
        
        # Update view matrices for all shaders
        view_matrix = self.view.astype('f4').tobytes()
        self.render_program['view'].write(view_matrix)
        
        # Set up additive blending for particles
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        
        # Render particles
        self.vao.render(moderngl.POINTS)
        
        # Set up alpha blending for circle
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        # Render reference circle
        
        # Swap buffers
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def cleanup(self):
        # Clean up ModernGL resources
        self.vertex_buffer.release()
        self.vao.release()
        self.render_program.release()
        self.ctx.release()
        glfw.terminate()

    def run(self):
        try:
            while not glfw.window_should_close(self.window):
                self.simulation.step()  
                self.render()
                
                if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                    break
        finally:
            self.cleanup()

if __name__ == "__main__":
    vis = GalaxyVisualization()
    print("Starting Galaxy Simulation...")
    vis.run()

