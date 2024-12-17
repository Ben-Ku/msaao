
struct Globals {
    mvp_transform: mat4x4<f32>,   
    mv_transform: mat4x4<f32>,   
    mv_rot: mat4x4<f32>,
    cam_pos: vec3<f32>,
    cam_dir: vec3<f32>,
};

var<uniform> globals: Globals;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) view_pos: vec3<f32>,
    @location(1) view_normal: vec3<f32>,
};

struct Vertex {
    ws_pos: vec3<f32>,
    ws_normal: vec3<f32>,
};

@vertex
fn vs_main(vertex: Vertex) -> VertexOutput {

    var vs_out: VertexOutput;
    vs_out.clip_pos = globals.mvp_transform * vec4(vertex.ws_pos, 1.0);
    vs_out.view_pos = (globals.mv_transform * vec4(vertex.ws_pos, 1.0)).xyz;
    vs_out.view_normal = (globals.mv_rot * vec4(vertex.ws_normal, 1.0)).xyz;

    return vs_out;
}

struct FragmentOutput {
    @location(0) view_pos: vec4<f32>,
    @location(1) view_normal: vec4<f32>
}


@fragment
fn fs_main(vs_out: VertexOutput) -> FragmentOutput {
    let view_pos = vec4(vs_out.view_pos, 0.0);

    let s = sign(abs(view_pos.z));
    let view_normal = vec4(vs_out.view_normal, s);


    return FragmentOutput(view_pos, view_normal);
}

