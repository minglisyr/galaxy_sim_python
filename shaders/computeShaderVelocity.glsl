#version 430

layout(local_size_x=16, local_size_y=16) in;

layout(rgba32f, binding=0) uniform image2D texturePosition;
layout(rgba32f, binding=1) uniform image2D textureVelocity;

// Simulation parameters
uniform float deltaTime;        // Time step
uniform float gravity;          // Gravitational constant
uniform float interactionRate;  // Percentage of particles to interact with
uniform float blackHoleForce;   // Force multiplier for central black hole
uniform float maxAcceleration;  // Maximum acceleration for color calculation

void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    int width = imageSize(texturePosition).x;
    int height = imageSize(texturePosition).y;
    
    // Skip if outside texture bounds
    if (pixel_coords.x >= width || pixel_coords.y >= height) {
        return;
    }

    // Read current particle data
    vec4 tmpPos = imageLoad(texturePosition, pixel_coords);
    vec3 pos = tmpPos.xyz;

    vec4 tmpVel = imageLoad(textureVelocity, pixel_coords);
    vec3 vel = tmpVel.xyz;
    float accColor = tmpVel.w;

    // Skip if this is an empty particle
    if (pos.x == 0.0 && pos.y == 0.0 && pos.z == 0.0) {
        return;
    }

    // Initialize acceleration
    vec3 acceleration = vec3(0.0);    // Calculate gravitational forces with subset of particles based on interaction rate
    int max_y = int(height * interactionRate);
    int max_x = int(width * interactionRate);
    
    for (int y = 0; y < max_y; y++) {
        for (int x = 0; x < max_x; x++) {
            ivec2 other_coords = ivec2(x, y);
            float idParticle2 = float(y * width + x);
            
            // Skip self interaction
            if (other_coords == pixel_coords) {
                continue;
            }

            // Get other particle data
            vec4 otherPos = imageLoad(texturePosition, other_coords);
            vec3 pos2 = otherPos.xyz;
            float isDarkMatter = otherPos.w;
            
            // Skip empty particles
            if (otherPos.x == 0.0 && otherPos.y == 0.0 && otherPos.z == 0.0) {
                continue;
            }            // Calculate the distance and displacement between particles
            vec3 dPos = pos2 - pos;
            float distance = length(dPos);

            // Calculate gravitational force with minimum distance to prevent singularity
            float distanceSq = (distance * distance) + 1.0;
            float gravityField = gravity / distanceSq;

            // Limit maximum acceleration from regular particles
            gravityField = min(gravityField, 1.0);

            // Apply stronger force for dark matter and black holes
            if (isDarkMatter > 0.5 || (pos2.x == 0.0 && pos2.y == 0.0 && pos2.z == 0.0)) {
                gravityField = gravity * blackHoleForce / distanceSq;
            }

            // Add the acceleration contribution
            acceleration += gravityField * normalize(dPos);
        }
    }

    // Update velocity based on acceleration
    vel += deltaTime * acceleration;

    // Calculate acceleration magnitude for visualization
    float acc_magnitude = length(acceleration);
    if (acc_magnitude > maxAcceleration) {
        accColor = maxAcceleration;
    } else {
        accColor = acc_magnitude;
    }

    // Store updated velocity and acceleration
    imageStore(textureVelocity, pixel_coords, vec4(vel, accColor));
}
