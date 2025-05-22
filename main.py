import moderngl
import numpy as np
import glfw        
import sys
import pyrr
from simulation import GalaxySimulation
from pathlib import Path

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

