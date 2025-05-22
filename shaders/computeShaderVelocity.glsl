#version 430

layout(local_size_x=16, local_size_y=16) in;

layout(rgba32f, binding=0) uniform image2D texturePosition;
layout(rgba32f, binding=1) uniform image2D textureVelocity;

// Simulation parameters
uniform float deltaTime;  // Time step
uniform float gravity;    // Gravitational constant
uniform float interactionRate;  // Percentage of particles to interact with
uniform float blackHoleForce;   // Extra force for dark matter
uniform float maxAcceleration;  // Maximum acceleration for color

void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    int width = imageSize(texturePosition).x;
    int height = imageSize(texturePosition).y;
    
    // Skip if outside texture bounds
    if (pixel_coords.x >= width || pixel_coords.y >= height) {
        return;
    }

    vec2 uv = vec2(pixel_coords) / vec2(width, height);
    float idParticle = uv.y * width + uv.x;

    // Read current particle data
    vec4 tmpPos = imageLoad(texturePosition, pixel_coords);
    vec3 pos = tmpPos.xyz;
    float isDarkMatter = tmpPos.w;

    vec4 tmpVel = imageLoad(textureVelocity, pixel_coords);
    vec3 vel = tmpVel.xyz;
    float accColor = tmpVel.w;

    // Skip if this is an empty particle
    if (pos.x == 0.0 && pos.y == 0.0 && pos.z == 0.0) {
        return;
    }

    // Initialize acceleration
    vec3 acceleration = vec3(0.0);

    // Calculate gravitational forces
    int interaction_width = int(width * interactionRate);
    int interaction_height = int(height * interactionRate);

    for (int y = 0; y < interaction_height; y++) {
        for (int x = 0; x < interaction_width; x++) {
            ivec2 other_coords = ivec2(x, y);
            float idParticle2 = float(y * width + x);

            if (idParticle == idParticle2) {
                continue;
            }

            vec4 pos2_data = imageLoad(texturePosition, other_coords);
            vec3 pos2 = pos2_data.xyz;
            float isDarkMatter2 = pos2_data.w;

            // Skip empty particles
            if (pos2.x == 0.0 && pos2.y == 0.0 && pos2.z == 0.0) {
                continue;
            }            vec3 diff = pos2 - pos;
            float distSqr = dot(diff, diff);
            distSqr = max(distSqr, 1.0);  // Minimum distance to prevent extreme forces
            
            // Calculate basic gravitational force
            float forceMagnitude = gravity / distSqr;
            forceMagnitude = min(forceMagnitude, 5.0);  // Limit maximum force
            
            // Add orbital motion component
            vec3 forceDir = normalize(diff);
            vec3 orbital = cross(normalize(pos), forceDir);  // Create perpendicular force for orbit
            
            // Combine forces: 70% towards center, 30% orbital
            acceleration += forceDir * forceMagnitude * 0.7 + orbital * forceMagnitude * 0.3;
            
            // Apply additional force if other particle is dark matter
            if (isDarkMatter2 > 0.5) {
                float blackHoleForceMag = min(blackHoleForce / distSqr, 10.0);
                acceleration += forceDir * blackHoleForceMag;
            }
            
            // Add central force for structure
            if (length(pos2) < 10.0) {
                float centralForceMag = min(blackHoleForce * 0.2, 8.0);
                acceleration += forceDir * centralForceMag;
            }
        }
    }    // Apply velocity damping
    vec3 newVel = vel * 0.9995 + acceleration * deltaTime;  // Reduced damping to 0.05% per frame
    
    // Apply orbital velocity correction
    vec3 radialDir = normalize(pos);
    vec3 currentOrbitalDir = cross(radialDir, normalize(vel));
    newVel += currentOrbitalDir * length(vel) * 0.01;  // Add slight orbital correction
    
    // Limit maximum velocity
    float speed = length(newVel);
    if (speed > 30.0) {  // Reduced maximum speed
        newVel = normalize(newVel) * 30.0;
    }
    
    // Calculate acceleration magnitude for coloring
    float accMag = length(acceleration);
    float newAccColor = mix(accColor, min(accMag / maxAcceleration, 1.0), 0.1);
    
    // Store updated velocity and acceleration magnitude
    imageStore(textureVelocity, pixel_coords, vec4(newVel, newAccColor));
}
