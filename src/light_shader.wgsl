
var depth_view: texture_depth_2d;
var depth_sampler: sampler;

var pos_view: texture_2d<f32>;
var pos_sampler: sampler;

var normal_view: texture_2d<f32>;
var normal_sampler: sampler;

var prev_ao_view: texture_2d<f32>;
var prev_ao_sampler: sampler;

var ao_view: texture_2d<f32>;
var ao_sampler: sampler;


var<uniform> globals: Globals;
var<uniform> ao_params: AOParams;


struct Globals {
    mvp_transform: mat4x4<f32>,   
    mv_transform: mat4x4<f32>,   
    mv_rot: mat4x4<f32>,
    cam_pos: vec3<f32>,
    cam_dir: vec3<f32>,
};


struct AOParams {
    is_first_pass: u32,
    is_last_pass: u32,
    ri_almost: f32,
    ao_width: f32,

    pad: vec3<u32>,
    ao_height: f32,
};

struct Vertex {
    ws_pos: vec3<f32>,
    ws_normal: vec3<f32>,
};


struct DownSampleOutput {
    @builtin(frag_depth) depth: f32,
    @location(0) pos: vec4<f32>,
    @location(1) normal: vec4<f32>
};


@fragment
fn fs_downsample(vertex: VertexOutput) -> DownSampleOutput {
    // let depth = textureSample(depth_view, depth_sampler, vertex.uv);
    // let subpixel_depths = textureGather(depth_view, depth_sampler, vertex.uv);
    let subpixel_px = textureGather(0 ,pos_view, pos_sampler, vertex.uv);
    let subpixel_py = textureGather(1 ,pos_view, pos_sampler, vertex.uv);
    // NOTE: using rh coordinate system means z values will be negative
    let subpixel_pz = -textureGather(2,pos_view, pos_sampler, vertex.uv);

    // NOTE: sorting algorithm
    // if num0 > num1: swap(num0,num1)
    // if num2 > num3: swap(num2,num3)
    // if num0 > num2: swap(num0,num2)
    // if num1 > num3: swap(num1,num3)
    // if num1 > num2: swap(num1,num2)
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
        if subpixel_pz[idx[i1]] > subpixel_pz[idx[i2]] {
            tmp = idx[i1];
            idx[i1] = idx[i2];
            idx[i2] = tmp;
        }
    }
    let p0z = subpixel_pz[idx[0]];
    let p1z = subpixel_pz[idx[1]];
    let p2z = subpixel_pz[idx[2]];
    let p3z = subpixel_pz[idx[3]];
    
    // NOTE: uncomment to verify pixels are sorted correctly
    // let sort_is_good = (d0 <= d1) && (d1 <= d2) && (d2 <= d3);
    // if !sort_is_good {
    //     c = 0.0;   
    // }

    
    var p_new: vec3<f32>;
    var n_new: vec3<f32>;

    let d_thresh = 1.0;

    let idx_1 = idx[1];

    
    let subpixel_nx = textureGather(0 ,normal_view, normal_sampler, vertex.uv);
    let subpixel_ny = textureGather(1 ,normal_view, normal_sampler, vertex.uv);
    let subpixel_nz = textureGather(2 ,normal_view, normal_sampler, vertex.uv);

    let p1 = vec3(subpixel_px[idx_1], subpixel_py[idx_1], subpixel_pz[idx_1]);
    let n1 = vec3(subpixel_nx[idx_1], subpixel_ny[idx_1], subpixel_nz[idx_1]);
    if p3z - p0z <= d_thresh {
        let idx_2 = idx[2];
        let p2 = vec3(subpixel_px[idx_2], subpixel_py[idx_2], subpixel_pz[idx_2]);
        let n2 = vec3(subpixel_nx[idx_2], subpixel_ny[idx_2], subpixel_nz[idx_2]);

        p_new = (p1 + p2) / 2.0;
        n_new = (n1 + n2) / 2.0;
    } else {
        p_new = p1; 
        n_new = n1; 
    }

    n_new = normalize(n_new);

    //NOTE: revert the negative z we added in the beginning
    p_new.z = -p_new.z;
    let output = DownSampleOutput(0.0, vec4(p_new, 1.0), vec4(vec3(n_new), 1.0));
    
    return output;
}


@fragment
fn fs_calc_ao(vertex: VertexOutput) -> @location(0) vec4<f32> {
    let view_pos = textureSample(pos_view, pos_sampler, vertex.uv).xyz;

    let r_max = 5.0;
    let d_max = 2.0;
    // NOTE: z is negative cause rh coordinate system
    let pz = -view_pos.z;
    let r_i = ao_params.ri_almost / pz; 

    // NOTE: kernel size
    // let R_i = floor(min(r_max, r_i));
    let R_i = 5.0;
    // let R_i = 5.0;

    let p = textureSample(pos_view, pos_sampler, vertex.uv).xyz;
    let n = textureSample(normal_view, normal_sampler, vertex.uv).xyz;

    let num_samples_x = u32(R_i) + 1;
    let N = f32(num_samples_x * num_samples_x);


    var dx = 1.0 / (ao_params.ao_width - 1.0);
    let dy = 1.0 / (ao_params.ao_height - 1.0);
    var sample_uv = vertex.uv - f32(R_i) * vec2(dx, dy);

    var ao_near = 0.0;
    for (var i: u32 = 0; i < num_samples_x; i++) {
        for (var j: u32 = 0; j < num_samples_x; j++) {
            let qi = textureSample(pos_view, pos_sampler, sample_uv).xyz;
            var d = (qi - p);
            let di = length(d);
            d /=  di;

            let rho = 1.0 - min(1.0, pow(di/d_max, 2.0));


            ao_near += 1.0 / N * rho * clamp(dot(n, d), 0.0, 1.0);
            sample_uv.x += 2.0 * dx;
        }
        sample_uv.x -= 2.0 * dx * f32(num_samples_x);
        sample_uv.y += 2.0 * dy;
    }

    var c = vec3(1.0 - ao_near);
    


    return vec4(c, 1.0);
}

@fragment
fn fs_light(vertex: VertexOutput) -> @location(0) vec4<f32> {
    let view_pos = textureSample(pos_view, pos_sampler, vertex.uv);
    let normal = textureSample(normal_view, normal_sampler, vertex.uv).xyz;

    let ws_normal = transpose(globals.mv_rot) * vec4(normal, 0.0);

    var depth = textureSample(depth_view, depth_sampler, vertex.uv);

    let sun_dir = normalize(vec3(0.5, -1.0, -0.8));
    // let ndotl = max(dot(normal.xyz, -sun_dir), 0.0);
    // let ndotl = max(dot(normal.xyz, -sun_dir), 0.0);

    let ndotl = max(dot(ws_normal.xyz, -sun_dir), 0.0);
    

    let ambient = 0.2;


    var c = vec3(ndotl);
    c += ambient;

    let ao = textureSample(ao_view, ao_sampler, vertex.uv);
    c = ao.xyz;

    // depth = linearize_depth(depth);
    // let a = -vec3(view_pos.z) / 10.0;

    // return vec4(a, 1.0);
    return vec4(c, 1.0);
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(vertex: Vertex) -> VertexOutput {
    // NOTE: ws pos equal to uv coords since vertex belongs to screen covering quad
    var uv = 0.5 * vertex.ws_pos.xy + 0.5;
    uv.y = 1.0 - uv.y;


    return VertexOutput(vec4(vertex.ws_pos,1.0), uv);
}


fn uv_2_texel(uv: vec2<f32>, wh: vec2<f32>) -> vec2<f32> {
    //NOTE: 0,0 corresponds to 0,0 and 1,1 corresponds to (w-1, h-1),
    // i.e actual indices
    return uv * (wh - 1.0);
}

fn linearize_depth(d: f32) -> f32
    {
        let zFar = 100.0;
        let zNear = 0.001;
        return zNear * zFar / (zFar + d * (zNear - zFar));
    }
