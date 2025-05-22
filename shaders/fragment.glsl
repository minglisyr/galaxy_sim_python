#version 430

in vec4 vColor;
out vec4 fragColor;

void main() {
    // Calculate distance from center of point
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    
    // Smooth circular points with soft edges
    float alpha = 1.0 - smoothstep(0.45, 0.5, dist);
    
    // Output final color with smooth alpha
    fragColor = vec4(vColor.rgb, vColor.a * alpha);
}