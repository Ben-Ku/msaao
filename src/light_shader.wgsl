struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

var depth_view: texture_depth_2d;
var depth_sampler: sampler;

var pos_view: texture_2d<f32>;
var pos_sampler: sampler;

var normal_view: texture_2d<f32>;
var normal_sampler: sampler;





struct Vertex {
    pos: vec3<f32>,
    normal: vec3<f32>,
};

fn linearize_depth(d: f32) -> f32
{
    let zFar = 100.0;
    let zNear = 0.001;
    return zNear * zFar / (zFar + d * (zNear - zFar));
}



struct DownSampleOutput {
    @builtin(frag_depth) depth: f32,
    @location(0) pos: vec4<f32>,
    @location(1) normal: vec4<f32>
};


@fragment
fn fs_downsample(vertex: VertexOutput) -> DownSampleOutput {
    let depth = textureSample(depth_view, depth_sampler, vertex.uv);
    let subpixel_depths = textureGather(depth_view, depth_sampler, vertex.uv);



    // if(num1>num2) swap(&num1,&num2);
    // if(num3>num4) swap(&num3,&num4);
    // if(num1>num3) swap(&num1,&num3);
    // if(num2>num4) swap(&num2,&num4);
    // if(num2>num3) swap(&num2,&num3);
    var idx = vec4<i32>(0,1,2,3);
    const idxs1 = array<i32,5>(0, 2, 0, 1, 1);
    const idxs2 = array<i32,5>(1, 3, 2, 3, 2);
    var i1: i32;
    var i2: i32;
    var tmp: i32;
    // sort indices in ascending order based on depth
    for (var i: i32 = 0; i < 5; i++) {
        i1 = idxs1[i];
        i2 = idxs2[i];
        if subpixel_depths[idx[i1]] > subpixel_depths[idx[i2]] {
            tmp = idx[i1];
            idx[i1] = idx[i2];
            idx[i2] = tmp;
        }
    }

    var c = depth;
    // let d0 = subpixel_depths[idx[0]];
    // let d1 = subpixel_depths[idx[1]];
    // let d2 = subpixel_depths[idx[2]];
    // let d3 = subpixel_depths[idx[3]];
    // let sort_is_good = (d0 <= d1) && (d1 <= d2) && (d2 <= d3);
    // if !sort_is_good {
    //     c = 0.0;   
    // }

    let output = DownSampleOutput(c, vec4(0.0), vec4(0.0));
    
    return output;
}

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


    var depth = textureSample(depth_view, depth_sampler, vertex.uv);

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
    depth = linearize_depth(depth);
    c = vec3(depth);
    return vec4(c / 10.0, 1.0);
}
