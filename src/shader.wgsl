
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

    // return VertexOutput(pos);
    // let vs_out = VertexOutput(pos,vec3(1.0,0.0,0.0) );
    // let vs_out = VertexOutput(pos);
    // let vs_out = VertexOutput(pos, vertex.normal);
    return vs_out;
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {

    let clip_pos = vertex.clip_pos;

    let depth = textureSampleCompare(depth_view, depth_sampler, vec2(0.5), 0.1);
    // let depth = textureSampleCompare(depth_view, depth_sampler, clip_pos.xy, 0.1);

    let sun_dir = normalize(vec3(0.5, -1.0, -0.8));

    let ndotl = max(dot(vertex.normal, -sun_dir), 0.0);

    let ambient = 0.2;

    var c = vec3(0.0);
    c += ndotl;
    c += ambient;

    c.x = depth;
    // c.x = h;

    return vec4(c,1.0);
}
