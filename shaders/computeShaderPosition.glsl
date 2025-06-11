#version 430

// Define the time step for the simulation
#define delta (1.0 / 30.0)

layout(binding = 0) uniform sampler2D texturePosition;
layout(binding = 1) uniform sampler2D textureVelocity;
layout(rgba32f, binding = 2) uniform image2D outputImage;

uniform vec2 resolution;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
    // Calculate the position in the output image
    ivec2 storePos = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = vec2(gl_GlobalInvocationID.xy) / resolution;

    vec4 tmpPos = texture(texturePosition, uv);
    vec3 pos = tmpPos.xyz;

    vec4 tmpVel = texture(textureVelocity, uv);
    vec3 vel = tmpVel.xyz;

    // Dynamics
    if (pos.x != 0.0 && pos.y != 0.0 && pos.z != 0.0) {
        pos += vel * delta;
    }

    // Output the positions and isDarkMatter in the output image
    imageStore(outputImage, storePos, vec4(pos, tmpPos.w));
}
