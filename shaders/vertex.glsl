#version 430

// Declare uniforms for texture samplers 
layout(binding=0) uniform sampler2D texturePosition;
layout(binding=1) uniform sampler2D textureVelocity;

// Declare uniforms for camera parameters and particle count
uniform float cameraConstant;
uniform float particlesCount;
uniform float uMaxAccelerationColor;
uniform float uLuminosity;
uniform float uHideDarkMatter;

// Declare output variable for color
out vec4 vColor;

// Normalize an acceleration value to a range of 0 to 1
float normalized(float acc){
    return (acc-0.)/(uMaxAccelerationColor-0.);
}

// If you use uv, it must be defined as an input or calculated
// If you don't have it in your original, you likely have to pass it as an attribute/in variable
// Assuming you have a vec2 uv input:
in vec2 uv;

void main() {
    // Retrieve position data from texture
    vec4 posTemp = texture(texturePosition, uv);
    vec3 pos = posTemp.xyz;
    float hideDarkMatter = posTemp.w;

    // Retrieve velocity data from texture and calculate acceleration
    vec4 velTemp = texture(textureVelocity, uv);
    vec3 vel = velTemp.xyz;
    float acc = velTemp.w;

    // In desktop GLSL, modelViewMatrix and projectionMatrix are not built-in; you must pass them as uniforms
    uniform mat4 modelViewMatrix;
    uniform mat4 projectionMatrix;
    vec4 mvPosition = modelViewMatrix * vec4( pos, 1.0 );

    /*** Size ***/
    gl_PointSize = 1.0;
    gl_PointSize *= ( 1.0 / - mvPosition.z );

    // Calculate the final position of the particle using the projection matrix
    gl_Position = projectionMatrix * mvPosition;

    /*** Color ***/
    vec3 hightAccelerationColor = vec3(1.,0.376,0.188);
    vec3 lowAccelerationColor = vec3(0.012,0.063,0.988);
    vec3 finalColor = vec3(0.0,0.0,0.0);
    if(uHideDarkMatter == 1.0) {
        if(hideDarkMatter == 0.0){
            finalColor = mix(lowAccelerationColor, hightAccelerationColor, normalized(acc));
        } else {
            finalColor = vec3(0.0,0.0,0.0);
        }
    } else {
        finalColor = mix(lowAccelerationColor, hightAccelerationColor, normalized(acc));
    }

    // Set the color of the particle
    vColor = vec4(finalColor, uLuminosity);
}
