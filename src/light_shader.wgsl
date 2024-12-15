var depth_view: texture_depth_2d;
var depth_sampler: sampler;

var pos_view: texture_2d<f32>;
var pos_sampler: sampler;

var normal_view: texture_2d<f32>;
var normal_sampler: sampler;

var ao_view: texture_2d<f32>;
var ao_sampler: sampler;


var prev_pos_view: texture_2d<f32>;
var prev_pos_sampler: sampler;

var prev_normal_view: texture_2d<f32>;
var prev_normal_sampler: sampler;

var prev_ao_view: texture_2d<f32>;
var prev_ao_sampler: sampler;




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
    let p = textureSample(pos_view, pos_sampler, vertex.uv).xyz;

    //NOTE: calc ao near
    let r_max = 5.0;
    let d_max = 2.0;
    // NOTE: z is negative cause rh coordinate system
    let pz = -p.z;
    let r_i = ao_params.ri_almost / pz; 

    // NOTE: kernel size
    let R_i = floor(min(r_max, r_i));
    // FIXME: uncomment above
    // let R_i = 5.0;

    let n = textureSample(normal_view, normal_sampler, vertex.uv).xyz;

    // FIXME: should be 0 samples sometimes
    let num_samples_x = u32(R_i) + 1;
    let N = f32(num_samples_x * num_samples_x);


    let dx = 1.0 / (ao_params.ao_width);
    let dy = 1.0 / (ao_params.ao_height);
    var sample_uv = vertex.uv - f32(R_i) * vec2(dx, dy);

    var ao_near_x = 0.0;
    for (var i: u32 = 0; i < num_samples_x; i++) {
        for (var j: u32 = 0; j < num_samples_x; j++) {
            let qi = textureSample(pos_view, pos_sampler, sample_uv).xyz;
            var d = (qi - p);
            let di = length(d);
            d /=  di;

            let rho = 1.0 - min(1.0, pow(di/d_max, 2.0));


            ao_near_x += rho * clamp(dot(n, d), 0.0, 1.0);
            sample_uv.x += 2.0 * dx;
        }
        sample_uv.x -= 2.0 * dx * f32(num_samples_x);
        sample_uv.y += 2.0 * dy;
    }

    let ao_near = vec2(ao_near_x, f32(N));


    // NOTE: CALC AO FAR

    // NOTE: calculate in which corner of the superpixel our current pixel is located to determine the
    // bilinear weights used to upsample, we are assuming the pixel is exactly twice as small, which is 
    // not necessarily true and depends on our output resolution
    // let w_vec =  1.0 / 16.0 * vec4(9.0, 3.0, 1.0, 3.0);
    let tmp = vertex.uv * vec2(ao_params.ao_width, ao_params.ao_height);
    let texel_coord = vec2<u32>(tmp - 0.5 + 0.0001);

    let uvb = texture_gather_weights(prev_ao_view,vertex.uv);
    // when doing a texure gather the values are laid out in the following order
    //    val3 val2
    //    val0 val1
    // so we need to set up weights correctly, our bilinear uv corresponds to the
    // relative position of val3 i.e value 3 should have weight (1 - x) * (1 - y)
    var w_bilinear: vec4f; 
    w_bilinear.x = (1.0-uvb.x) * uvb.y;
    w_bilinear.y = uvb.x * uvb.y;
    w_bilinear.z = uvb.x * (1.0-uvb.y);
    w_bilinear.w = (1.0-uvb.x) * (1.0-uvb.y);


    let superpixel_nx = textureGather(0, prev_normal_view, prev_normal_sampler, vertex.uv);
    let superpixel_ny = textureGather(1, prev_normal_view, prev_normal_sampler, vertex.uv);
    let superpixel_nz = textureGather(2, prev_normal_view, prev_normal_sampler, vertex.uv);


    var w_normal: vec4f;
    // QUESTION: better to write loop or better to have it all written out
    const tn = 8.0;
    for (var i: u32 = 0; i < 4; i ++) {
        let ndot = dot(n, vec3(superpixel_nx[i], superpixel_ny[i], superpixel_nz[i]));
        w_normal[i] = pow(0.5*ndot + 0.5, tn);
    }

    let superpixel_z = -textureGather(2, prev_pos_view, prev_pos_sampler, vertex.uv);
    var w_depth: vec4f;
    const tz = 16.0;
    for (var i: u32 = 0; i < 4; i ++) {
        w_depth[i] = pow(1.0 / (1.0 + abs(superpixel_z[i] - pz)), tz);
    }

    var w_bilateral = w_bilinear * w_normal * w_depth;
    // w_bilateral = w_depth;

    let superpixel_ao0 = textureGather(0, prev_ao_view, prev_ao_sampler, vertex.uv);
    let superpixel_ao1 = textureGather(1, prev_ao_view, prev_ao_sampler, vertex.uv);
    let superpixel_ao2 = textureGather(2, prev_ao_view, prev_ao_sampler, vertex.uv);
    // QUESTION: should it be normalized or not
    // var ao_far = dot(w_bilateral, superpixel_ao) / w_tot;
    var ao_far: vec3f;
    ao_far[0] = dot(w_bilateral, superpixel_ao0);
    ao_far[1] = dot(w_bilateral, superpixel_ao1);
    ao_far[2] = dot(w_bilateral, superpixel_ao2);
    let w_tot = dot(w_bilateral, vec4(1.0));
    // ao_far[2] /= w_tot ;
    

    var c: vec3f;
    if ao_params.is_first_pass == 1 {
        var res: vec3f;
        res[0] = ao_near[0] / ao_near[1];
        res[1] = ao_near[1];
        return vec4(res,1.0);
    } else {
        var ao_comb: vec3f;
        ao_comb[0] = max(ao_near[0] / ao_near[1], ao_far[0]);
        ao_comb[1] = ao_near[0] + ao_far[1];
        ao_comb[2] = ao_near[1] + ao_far[2];

        c = ao_comb;
    } 

    if ao_params.is_last_pass == 1 {
        let ao_max = max(ao_near[0] / ao_near[1], ao_far[0]);
        let ao_avg = (ao_far[1] + ao_near[0]) / (ao_far[2] + ao_near[1]);
        let ao_final = 1.0 - (1.0 - ao_max) * (1.0 - ao_avg);
        c = vec3(ao_final);
    }
    return vec4(c, 1.0);
}

@fragment
fn fs_blur_ao(vertex: VertexOutput) -> @location(0) vec4<f32> {

    
    // let ao = textureSample(ao_view, ao_sampler, vertex.uv).xyz;

    let dx = 1.0 / (ao_params.ao_width);
    let dy = 1.0 / (ao_params.ao_height);

    // NOTE: actual gaussian kernel should be something like 
    // https://stackoverflow.com/questions/20746172/blur-an-image-using-3x3-gaussian-kernel
    // these weights dont sum to 1 on 3x3 kernel but its what the paper said...
    // let weights =  vec3(0.25, 0.5, 1.0);
    // NOTE: use weights that sum to 1 over 3x3 
    const weights = 0.25  * vec3(0.25, 0.5, 1.0);

    var uv = vertex.uv - vec2(dx,dy);
    var ao_blur = vec3(0.0);
    for (var i: u32 = 0; i < 3; i++) {
        for (var j: u32 = 0; j < 3; j++) {
            // NOTE: will be 0 in corners, 1 on middle sides and 2 in middle
            let weight_i = (i % 2) + (j % 2); 
            let w = weights[weight_i];
            let ao = textureSample(ao_view, ao_sampler, uv).xyz;

            ao_blur += w * ao;
            uv.x += dx;
        }
        uv.x -= 3.0 * dx;
        uv.y += dy;
    }


    // let uvx = vertex.uv + 0.5 * vec2(dx,dy);


    // let box = textureGather(0,ao_view, ao_sampler, vertex.uv);
    // let boy = textureGather(1,ao_view, ao_sampler, vertex.uv);
    // let boz = textureGather(2,ao_view, ao_sampler, vertex.uv);
    // ao_blur = vec3(box.w,boy.w,boz.w);

    return vec4(ao_blur, 1.0);
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


    let ao_max = ao[0];
    let ao_avg = ao[1] / ao[2];
    // let ao_final = 1.0 - (1.0 - ao_max) * (1.0 - ao_avg);
    let ao_final = ao[0];
    c = vec3(1.0 - ao_final);
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

// from https://github.com/eliemichel/WebGPU-utils/blob/main/wgsl/textureGatherWeights.wgsl
fn texture_gather_weights(t: texture_2d<f32>, coords: vec2f) -> vec2f {
    let dim = textureDimensions(t).xy;
    let scaled_uv = coords * vec2f(dim);
    // This is not accurate, see see https://www.reedbeta.com/blog/texture-gathers-and-coordinate-precision/
    // but bottom line is:
    //   "Unfortunately, if we need this to work, there seems to be no option but to check
    //    which hardware you are running on and apply the offset or not accordingly."
    // return fract(scaled_uv - 0.5 + 1.0 / 512.0);
    return fract(scaled_uv - 0.5);
}
