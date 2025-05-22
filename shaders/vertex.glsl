#version 430

layout (location = 0) in vec4 position;  // w component is dark matter flag
layout (location = 1) in vec4 color;     // w component is brightness

uniform mat4 projection;
uniform mat4 view;
uniform float pointSize;
uniform float brightness;

out vec4 vColor;

void main() {
    gl_Position = projection * view * vec4(position.xyz, 1.0);
    
    // Adjust point size based on distance from camera
    float dist = length(gl_Position.xyz);
    gl_PointSize = pointSize * (position.w > 0.5 ? 3.0 : 1.0) / dist;
    
    // Adjust color brightness and handle dark matter
    vColor = color;
    if (position.w > 0.5) {
        // Black hole/dark matter
        vColor = vec4(1.0, 0.2, 0.0, 1.0);  // Orange-red for black hole
    }
    vColor.rgb *= brightness;  // Apply brightness adjustment
}