#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

struct PhysicsVertex {
    vec4 pos;
    vec4 vel;
    vec4 prevPos;
    vec4 normal;
    float invMass;
    float isFixed;
    float stress;
    float padding;
    vec4 colData;
};

struct RenderVertex {
    vec4 color;
};

layout(std430, binding = 1) readonly buffer PhysicsBuffer {
    PhysicsVertex physicsVerts[];
};

layout(std430, binding = 2) readonly buffer RenderBuffer {
    RenderVertex renderVerts[];
};
layout(location = 0) out vec3 fragColor;

void main() {
    PhysicsVertex p = physicsVerts[gl_VertexIndex];
    RenderVertex r = renderVerts[gl_VertexIndex];

    gl_PointSize = 3.0;
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(p.pos.xyz, 1.0);
    fragColor = r.color.xyz;
}
