import moderngl
import numpy as np
import glfw
import sys
import pyrr
from simulation import GalaxySimulation
from text_renderer import TextRenderer

class GalaxyVisualization:
    def __init__(self, width=1920, height=1080):
        # Make sure GLFW hasn't been initialized yet
        try:
            glfw.terminate()
        except:
            pass

        # Initialize GLFW
        if not glfw.init():
            sys.exit("Could not initialize GLFW")

        # Configure GLFW
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, 4)  # 4x MSAA
        glfw.window_hint(glfw.VISIBLE, True)

        # Create window
        self.window = glfw.create_window(width, height, "Galaxy Simulation", None, None)
        if not self.window:
            glfw.terminate()
            sys.exit("Could not create window")

        # Set up OpenGL context
        glfw.make_context_current(self.window)
        self.ctx = moderngl.create_context()
        
        # Enable necessary OpenGL features
        self.ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self.ctx.point_size = 1.0  # Base point size (will be modified by shader)

        # Create simulation
        self.simulation = GalaxySimulation()

        # Set up camera
        self.camera_distance = 400000.0  # Increased to see the whole galaxy
        self.camera_rotation = pyrr.euler.create(0.5, 0.3, 0)  # Angled view
        self.last_mouse = None
        self.setup_camera()

        # Create particle rendering program
        self.setup_particle_rendering()
        
        # Create text renderer
        self.text_renderer = TextRenderer(self.ctx, width, height)
        
        # Store window dimensions
        self.width = width
        self.height = height
        
        # Setup help text
        self.help_text = [
            "Controls:",
            "Left Mouse: Rotate camera",
            "Right Mouse: Pan camera",
            "Scroll: Zoom in/out",
            "ESC: Exit"
        ]

        # Set up input callbacks
        glfw.set_mouse_button_callback(self.window, self.handle_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self.handle_mouse_move)
        glfw.set_scroll_callback(self.window, self.handle_scroll)

    def setup_camera(self):
        self.proj = pyrr.matrix44.create_perspective_projection(
            45.0, self.window_ratio, 1.0, 1000.0  # Adjusted near/far planes for scale
        )
        self.update_view_matrix()

    def update_view_matrix(self):
        # Create view matrix
        rotation = pyrr.matrix44.create_from_eulers(self.camera_rotation)
        translation = pyrr.matrix44.create_from_translation([0, 0, -self.camera_distance])
        self.view = pyrr.matrix44.multiply(rotation, translation)

    @property
    def window_ratio(self):
        width, height = glfw.get_window_size(self.window)
        return width / height

    def setup_particle_rendering(self):
        try:
            # Create shaders for particle rendering
            vertex_shader = '''
                #version 430
                
                uniform mat4 projection;
                uniform mat4 view;
                uniform float point_size;
                
                layout(location = 0) in vec3 in_position;
                layout(location = 1) in vec4 in_color;
                
                out vec4 v_color;
                
                void main() {
                    vec4 viewPos = view * vec4(in_position, 1.0);
                    gl_Position = projection * viewPos;
                    
                    // Calculate screen-space point size
                    float dist = length(viewPos.xyz);
                    gl_PointSize = point_size / (1.0 + dist * 0.1);
                    
                    // Adjust color based on distance
                    v_color = in_color;
                    v_color.a *= clamp(2.0 - dist * 0.005, 0.0, 1.0);  // Fade out distant particles
                }
            '''

            fragment_shader = '''
                #version 430
                
                in vec4 v_color;
                out vec4 f_color;
                
                void main() {
                    // Create circular points with glow
                    vec2 coord = gl_PointCoord * 2.0 - 1.0;
                    float r = dot(coord, coord);
                    if (r > 1.0) discard;
                    
                    // Apply soft edges with emission
                    float alpha = 1.0 - smoothstep(0.5, 1.0, r);
                    vec3 emissive = v_color.rgb * (1.0 + 2.0 * (1.0 - r));  // Add glow
                    f_color = vec4(emissive, v_color.a * alpha);
                }
            '''

            self.render_program = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader
            )
            print("\nShader compilation successful!")

            # Set up uniforms
            self.render_program['projection'].write(self.proj.astype('f4').tobytes())
            self.render_program['point_size'].value = 500.0  # Much larger for better visibility
            
            # Debug print uniform values
            print("\nUniform values:")
            print(f"Projection matrix:\n{self.proj}")
            print(f"Initial view matrix:\n{self.view}")
            print(f"Point size: {self.render_program['point_size'].value}")

            # Create vertex buffer for particles
            self.vertex_buffer = self.ctx.buffer(reserve=self.simulation.num_stars * 28)  # pos(12) + color(16)
            self.vao = self.ctx.vertex_array(
                self.render_program,
                [
                    (self.vertex_buffer, '3f 4f', 'in_position', 'in_color')
                ]
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

    def handle_mouse_button(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.last_mouse = glfw.get_cursor_pos(window)
            else:
                self.last_mouse = None

    def handle_mouse_move(self, window, xpos, ypos):
        if self.last_mouse is not None:
            dx = xpos - self.last_mouse[0]
            dy = ypos - self.last_mouse[1]
            
            # Smoother rotation with adjusted sensitivity
            rotation_speed = 0.003
            self.camera_rotation[1] += dx * rotation_speed
            self.camera_rotation[0] = max(min(
                self.camera_rotation[0] + dy * rotation_speed,
                np.pi/2 * 0.8  # Limit vertical rotation to prevent gimbal lock
            ), -np.pi/2 * 0.8)
            
            self.update_view_matrix()
            self.last_mouse = (xpos, ypos)

    def handle_scroll(self, window, xoffset, yoffset):
        # Smoother zoom with smaller steps
        zoom_factor = 0.95 if yoffset > 0 else 1.05
        self.camera_distance = max(50.0, min(
            self.camera_distance * zoom_factor,
            500.0
        ))
        self.update_view_matrix()

    def render(self):
        # Clear screen
        self.ctx.clear(0.0, 0.0, 0.05, 1.0)  # Dark blue background
        
        # Update view matrix uniform
        self.render_program['view'].write(self.view.astype('f4').tobytes())
        
        # Get updated particle data
        positions, colors = self.simulation.get_particle_data()
        
        # Update vertex buffer with new particle data
        vertex_data = np.zeros(self.simulation.num_stars * 7, dtype='f4')
        # Interleave position and color data
        for i in range(self.simulation.num_stars):
            idx = i * 7
            vertex_data[idx:idx+3] = positions[i][:3]  # Only take xyz components
            vertex_data[idx+3:idx+7] = colors[i]   # rgba
        self.vertex_buffer.write(vertex_data.tobytes())
        
        # Render particles
        self.vao.render(moderngl.POINTS)
        
        # Render text overlay
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        # Render help text
        y_offset = 20
        for line in self.help_text:
            self.text_renderer.render_text(line, (20, y_offset), color=(1, 1, 1, 0.8))
            y_offset += 30
        
        # Render particle count
        count_text = f"Particles: {self.simulation.num_stars}"
        self.text_renderer.render_text(count_text, (20, self.height - 40), color=(0.8, 0.8, 1.0, 0.8))
        
        # Reset blend function for particles
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        
        # Swap buffers
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def run(self):
        while not glfw.window_should_close(self.window):
            self.simulation.step()  
            self.render()
            
            # Handle escape key
            if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break
        
        glfw.terminate()

if __name__ == "__main__":
    try:
        # Create and run the visualization
        vis = GalaxyVisualization()
        print("Starting Galaxy Simulation...")
        vis.run()
    except Exception as e:
        print(f"Error running simulation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
