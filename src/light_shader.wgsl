struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

var pos_view: texture_2d<f32>;
var pos_sampler: sampler;

var normal_view: texture_2d<f32>;
var normal_sampler: sampler;


struct Vertex {
    pos: vec3<f32>,
    normal: vec3<f32>,
};


@vertex
fn vs_main(vertex: Vertex) -> VertexOutput {
    var uv = 0.5*vertex.pos.xy + 0.5;
    uv.y = 1.0 - uv.y;

    return VertexOutput(vec4(vertex.pos,1.0), uv);
}
@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {

    let ws_pos = textureSample(pos_view, pos_sampler, vertex.uv);
    let normal = textureSample(normal_view, normal_sampler, vertex.uv);


    let sun_dir = normalize(vec3(0.5, -1.0, -0.8));
    let ndotl = max(dot(normal.xyz, -sun_dir), 0.0);
    

    let ambient = 0.2;
    // let pos = vertex.pos;
    // let pos_01 = 0.5*pos + 0.5;

    // let d = pos.length()
    var r = 1.0;
    var b = 0.0;

    // if pos_01.x > 100.0 {
    //     b = 1.0;
    //     r = 0.0;
    // }

    var c = vec3(ndotl);
    c += ambient;
    return vec4(c, 1.0);
    // return vec4(pos_01.xy,0.0,1.0);
}
