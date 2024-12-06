
struct Globals {
    mvp_transform: mat4x4<f32>,   
};

var<uniform> globals: Globals;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) normal: vec3<f32>
};

struct Vertex {
    pos: vec3<f32>,
    normal: vec3<f32>,
};

@vertex
fn vs_main(vertex: Vertex) -> VertexOutput {
    var pos = vec4(vertex.pos, 1.0);

    pos = globals.mvp_transform * pos;

    // return VertexOutput(pos);
    let vs_out = VertexOutput(pos, vertex.normal);
    // let vs_out = VertexOutput(pos,vec3(1.0,0.0,0.0) );
    // let vs_out = VertexOutput(pos);
    return vs_out;
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {

    // var light_dir = vec3(1.0,2.0,3.0);
    let sun_dir = normalize(vec3(0.5, -1.0, -0.8));
    // light_dir = normalize(light_dir);

    let ndotl = max(dot(vertex.normal, -sun_dir), 0.0);
    // let d = abs(dot(vertex.normal, sun_dir));
    // let d = vertex.normal.x;
    // return vec4(d,d,d,1.0);
    // return vec4(1.0,0.0,0.0,1.0);
    // return vec4(d,0.0,0.0,1.0);
    // let depth = vertex.clip_pos.z / vertex.clip_pos.w / 50.0; 
    // return vec4(depth,0.0,0.0,1.0);
    // return vec4(d,0.0,0.0,1.0);

    let ambient = 0.2;

    var c = vec3(0.0);
    c += ndotl;
    c += ambient;

    return vec4(c,1.0);
}
