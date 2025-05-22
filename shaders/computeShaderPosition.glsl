#version 430

layout(local_size_x=16, local_size_y=16) in;

layout(rgba32f, binding=0) uniform image2D texturePosition;
layout(rgba32f, binding=1) uniform image2D textureVelocity;

uniform float deltaTime;

void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
      // Get texture dimensions
    int width = imageSize(texturePosition).x;
    int height = imageSize(texturePosition).y;
    
    // Skip if outside texture bounds
    if (pixel_coords.x >= width || pixel_coords.y >= height) {
        return;
    }    // Read current position and velocity
    vec4 tmpPos = imageLoad(texturePosition, pixel_coords);
    vec3 pos = tmpPos.xyz;
    float isDarkMatter = tmpPos.w;

    vec4 tmpVel = imageLoad(textureVelocity, pixel_coords);
    vec3 vel = tmpVel.xyz;

    // Update position using velocity (skip central black hole)
    if (isDarkMatter < 0.5 && (pos.x != 0.0 || pos.y != 0.0 || pos.z != 0.0)) {
        pos += vel * deltaTime;
    }

    // Store updated position, preserving dark matter flag
    imageStore(texturePosition, pixel_coords, vec4(pos, isDarkMatter));
}
