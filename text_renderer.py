import numpy as np
from PIL import Image, ImageDraw, ImageFont
import moderngl

class TextRenderer:
    def __init__(self, ctx, width, height):
        self.ctx = ctx
        self.width = width
        self.height = height
        
        # Create text rendering program
        self.setup_program()
        
        # Load or create a font
        try:
            self.font = ImageFont.truetype("arial.ttf", 24)
        except:
            self.font = ImageFont.load_default()
            
    def setup_program(self):
        vertex_shader = '''
            #version 430
            
            layout(location = 0) in vec2 in_position;
            layout(location = 1) in vec2 in_texcoord;
            
            uniform vec2 screen_size;
            uniform vec2 position;
            uniform vec2 scale;
            
            out vec2 v_texcoord;
            
            void main() {
                vec2 screen_pos = ((in_position * scale + position) / screen_size) * 2.0 - 1.0;
                gl_Position = vec4(screen_pos, 0.0, 1.0);
                v_texcoord = in_texcoord;
            }
        '''

        fragment_shader = '''
            #version 430
            
            uniform sampler2D text_texture;
            uniform vec4 text_color;
            
            in vec2 v_texcoord;
            out vec4 f_color;
            
            void main() {
                float alpha = texture(text_texture, v_texcoord).r;
                f_color = text_color * vec4(1.0, 1.0, 1.0, alpha);
            }
        '''

        self.program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
        
        # Create a quad for rendering text
        vertices = np.array([
            # positions    # texture coords
            0.0, 0.0,     0.0, 1.0,
            1.0, 0.0,     1.0, 1.0,
            0.0, 1.0,     0.0, 0.0,
            1.0, 1.0,     1.0, 0.0,
        ], dtype='f4')
        
        self.vertex_buffer = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (self.vertex_buffer, '2f 2f', 'in_position', 'in_texcoord')
            ]
        )
    
    def create_text_texture(self, text):
        # Measure text size
        bbox = self.font.getbbox(text)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        # Create image with padding
        img = Image.new('L', (w + 4, h + 4), color=0)
        draw = ImageDraw.Draw(img)
        
        # Draw text
        draw.text((2, 2), text, font=self.font, fill=255)
        
        # Create texture
        texture = self.ctx.texture(img.size, 1, img.tobytes())
        texture.use(0)
        
        return texture, img.size
        
    def render_text(self, text, position, color=(1, 1, 1, 1)):
        # Create texture for text
        texture, size = self.create_text_texture(text)
        
        # Set uniforms
        self.program['screen_size'].value = (float(self.width), float(self.height))
        self.program['position'].value = position
        self.program['scale'].value = size
        self.program['text_color'].value = color
        self.program['text_texture'].value = 0
        
        # Draw text
        texture.use(0)
        self.vao.render(moderngl.TRIANGLE_STRIP)
        
        # Clean up
        texture.release()
