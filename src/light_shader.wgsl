struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) pos: vec2<f32>,
};


struct Vertex {
    pos: vec3<f32>,
    normal: vec3<f32>,
};


@vertex
fn vs_main(vertex: Vertex) -> VertexOutput {
    return VertexOutput(vec4(vertex.pos,1.0), vertex.pos.xy);
}
@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    // let pos = vertex.clip_pos;
    let pos = vertex.pos;
    let pos_01 = 0.5*pos + 0.5;

    // let d = pos.length()
    var r = 1.0;
    var b = 0.0;

    if pos_01.x > 100.0 {
        b = 1.0;
        r = 0.0;
    }

    // return vec4(r,0.0,b,1.0);
    return vec4(pos_01.xy,0.0,1.0);
}
