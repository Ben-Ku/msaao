
struct Globals {
    mvp_transform: mat4x4<f32>,   
    cam_pos: vec3<f32>,
    cam_dir: vec3<f32>,
};

var<uniform> globals: Globals;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) ws_pos: vec3<f32>,
};

var depth_view: texture_depth_2d;
var depth_sampler: sampler_comparison;

struct Vertex {
    pos: vec3<f32>,
    normal: vec3<f32>,
};

@vertex
fn vs_main(vertex: Vertex) -> VertexOutput {

    var vs_out: VertexOutput;
    vs_out.ws_pos = vertex.pos;
    vs_out.clip_pos = globals.mvp_transform * vec4(vertex.pos, 1.0);;
    vs_out.normal = vertex.normal;

    return vs_out;
}

struct FragmentOutput {
    @location(0) ws_pos : vec4<f32>,
    @location(1) normal : vec4<f32>
}


@fragment
fn fs_main(vertex: VertexOutput) -> FragmentOutput {
    let clip_pos = vertex.clip_pos;
    let sun_dir = normalize(vec3(0.5, -1.0, -0.8));
    let ndotl = max(dot(vertex.normal, -sun_dir), 0.0);
    let ambient = 0.2;
    var c = vec3(0.0);
    c += ndotl;
    c += ambient;
    // c.x = depth;
    // c.x = h;



    // let ws_pos = vec4(vertex.ws_pos,clip_pos.z);
    let ws_pos = vec4(vertex.ws_pos,clip_pos.z);

    let normal = vec4(vertex.normal, 0.0);


    return FragmentOutput(ws_pos, normal);
    // return vec4(c,1.0);
}

