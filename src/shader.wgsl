
struct Globals {
    mvp_transform: mat4x4<f32>,   
};

var<uniform> globals: Globals;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

struct Vertex {
    pos: vec3<f32>,
};

@vertex
fn vs_main(vertex: Vertex) -> VertexOutput {
    var pos = vec4(vertex.pos, 1.0);
    pos = globals.mvp_transform * pos;
    return VertexOutput(pos);
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    return vec4(1.0,0.0,0.0,1.0);
}
