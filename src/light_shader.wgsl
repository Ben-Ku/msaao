
// struct BasicVertex {
//     pos: vec3<f32>,
// };


struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    // @location(0) normal: vec3<f32>,
    // @location(1) ws_pos: vec3<f32>,
};


struct Vertex {
    pos: vec3<f32>,
    normal: vec3<f32>,
};
// var depth_view: texture_depth_2d;
// var depth_sampler: sampler_comparison;


@vertex
fn vs_main(vertex: Vertex) -> VertexOutput {
    // var vs_out: VertexOutput;
    // vs_out.ws_pos = vertex.pos;
    // vs_out.clip_pos = globals.mvp_transform * vec4(vertex.pos, 1.0);;
    // vs_out.normal = vertex.normal;

    // return VertexOutput(pos);
    // let vs_out = VertexOutput(pos,vec3(1.0,0.0,0.0) );
    // let vs_out = VertexOutput(pos);
    // let vs_out = VertexOutput(pos, vertex.normal);
    return VertexOutput(vec4(vertex.pos,1.0));
}
@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    let pos = vertex.clip_pos;
    let pos_01 = 0.5*pos + 0.5;

    // let d = pos.length()
    var r = 1.0;
    var b = 0.0;

    if pos_01.x > 100.0 {
        b = 1.0;
        r = 0.0;
    }

    return vec4(r,0.0,b,1.0);
}
