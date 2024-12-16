use std::io::{BufRead, Read, Write};

use nanorand::Rng;

pub use blade_graphics as gpu;
use bytemuck::{Pod, Zeroable};
pub use glam::*;

pub const PI: f32 = 3.14159265358979323846264338327950288;
pub const TAU: f32 = 2.0 * PI;

pub const NUM_AO_TEXTURES: usize = 5;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Globals {
    mvp_transform: [[f32; 4]; 4],
    mv_transform: [[f32; 4]; 4],
    mv_rot: [[f32; 4]; 4],
    cam_pos: [f32; 3],
    cam_dir: [f32; 3],
    pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct AOParams {
    pub num_passes: u32,
    pub pass_i: u32,
    pub ri_almost: f32,
    pub ao_width: f32,

    pub pad: [u32; 3],
    pub ao_height: f32,
}

impl AOParams {
    pub fn from(pass_i: usize, d_max: f32, fov_y: f32, ao_width: u32, ao_height: u32) -> Self {
        let s = ao_height as f32;
        let a = fov_y;
        // NOTE: eq (5) in reference paper, we need to divide by depth of specific pixel in shader tog get exact value
        let r0_almost = s * d_max / (2.0 * (a / 2.0).tan());
        let ri_almost = r0_almost / ((1 << pass_i) as f32);
        let ri_almost = ri_almost;
        Self {
            ri_almost,
            pad: Default::default(),
            ao_width: ao_width as f32,
            ao_height: ao_height as f32,
            num_passes: NUM_AO_TEXTURES as _,
            pass_i: pass_i as _,
        }
    }
}

#[derive(blade_macros::ShaderData)]
pub struct GeometryParams {
    pub globals: Globals,
}

// #[derive(blade_macros::ShaderData)]
// pub struct LightParams {
//     pub pos_view: gpu::TextureView,
//     pub pos_sampler: gpu::Sampler,

//     pub normal_view: gpu::TextureView,
//     pub normal_sampler: gpu::Sampler,

//     pub depth_view: gpu::TextureView,
//     pub depth_sampler: gpu::Sampler,
// }

#[derive(blade_macros::ShaderData)]
pub struct DepthPosNormalParams {
    pub globals: Globals,
    pub depth_view: gpu::TextureView,
    pub depth_sampler: gpu::Sampler,

    pub pos_view: gpu::TextureView,
    pub pos_sampler: gpu::Sampler,

    pub normal_view: gpu::TextureView,
    pub normal_sampler: gpu::Sampler,
}

#[derive(blade_macros::ShaderData)]
pub struct BlurParams {
    pub ao_params: AOParams,

    pub ao_view: gpu::TextureView,
    pub ao_sampler: gpu::Sampler,
}

#[derive(blade_macros::ShaderData)]
pub struct LightPassParams {
    pub globals: Globals,
    pub depth_view: gpu::TextureView,
    pub depth_sampler: gpu::Sampler,

    pub pos_view: gpu::TextureView,
    pub pos_sampler: gpu::Sampler,

    pub normal_view: gpu::TextureView,
    pub normal_sampler: gpu::Sampler,

    pub ao_view: gpu::TextureView,
    pub ao_sampler: gpu::Sampler,
}

#[derive(blade_macros::ShaderData)]
pub struct CalcAoParams {
    pub pos_view: gpu::TextureView,
    pub pos_sampler: gpu::Sampler,

    pub normal_view: gpu::TextureView,
    pub normal_sampler: gpu::Sampler,

    pub prev_pos_view: gpu::TextureView,
    pub prev_pos_sampler: gpu::Sampler,

    pub prev_normal_view: gpu::TextureView,
    pub prev_normal_sampler: gpu::Sampler,

    pub prev_ao_view: gpu::TextureView,
    pub prev_ao_sampler: gpu::Sampler,

    pub ao_params: AOParams,
}

#[derive(blade_macros::ShaderData)]
pub struct PosNormalPrevAOParams {
    pub globals: Globals,
    pub pos_view: gpu::TextureView,
    pub pos_sampler: gpu::Sampler,

    pub normal_view: gpu::TextureView,
    pub normal_sampler: gpu::Sampler,

    pub prev_ao_view: gpu::TextureView,
    pub prev_ao_sampler: gpu::Sampler,

    pub ao_params: AOParams,
    // pub is_first_pass: u32,
    // pub pad: [u32; 3],
}

// #[derive(blade_macros::ShaderData)]
// pub struct DepthDownsampleParams {
//     pub depth: gpu::TextureView,
//     pub depth_from_sampler: gpu::Sampler,
// }

#[derive(blade_macros::Vertex, Debug)]
pub struct Vertex {
    pub ws_pos: [f32; 3],
    pub ws_normal: [f32; 3],
}

pub struct Mesh {
    pub vertex_buf: gpu::BufferPiece,
    pub index_buf: Option<gpu::BufferPiece>,
    pub num_vertices: usize,
    pub num_indices: usize,
}

pub struct CpuMesh {
    pub vertices: Vec<Vec3A>,
    pub indices: Vec<usize>,
}

#[derive(Clone)]
pub struct Camera {
    pub pos: Vec3A,
    pub yaw: f32,
    pub pitch: f32,
    pub vfov_rad: f32,
    pub aspect: f32,
}

pub struct InputState {
    pub ao_level: usize,
    pub use_blur: bool,
}

pub struct GBuffer {
    pub depth_textures: DepthTextures,
    pub pos_texture: gpu::Texture,
    pub normal_texture: gpu::Texture,

    pub pos_view: gpu::TextureView,
    pub normal_view: gpu::TextureView,

    pub pos_sampler: gpu::Sampler,
    pub normal_sampler: gpu::Sampler,
}

pub struct TextureStuff {
    pub texture: gpu::Texture,
    pub view: gpu::TextureView,
    pub sampler: gpu::Sampler,
    pub size: gpu::Extent,
}

pub struct DepthPosNormalTexture {
    pub depth: TextureStuff,
    pub pos: TextureStuff,
    pub normal: TextureStuff,
}

pub struct DownsampleTextures {
    pub textures: Vec<DepthPosNormalTexture>,
}

pub struct AOTextures {
    pub textures: Vec<TextureStuff>,
    pub textures_after_blur: Vec<TextureStuff>,
    pub dummy_texture: TextureStuff,
}

pub struct DepthTextures {
    pub texture_stuffs: Vec<TextureStuff>,
}

pub fn create_downsample_and_ao_textures(
    ctx: &gpu::Context,
    screen_size: gpu::Extent,
) -> (DownsampleTextures, AOTextures) {
    let mut depth_pos_normal_textures = vec![];
    let mut ao_textures = vec![];
    let mut ao_textures_blur = vec![];

    let width = screen_size.width;
    let height = screen_size.height;

    for i in 0..NUM_AO_TEXTURES {
        let pow_2 = 1 << i;
        let width_i = width / pow_2;
        let height_i = height / pow_2;

        let extent_i = gpu::Extent {
            width: width_i,
            height: height_i,
            depth: 1,
        };

        let depth_texture_i = ctx.create_texture(gpu::TextureDesc {
            name: format!("depth texture {i}").as_str(),
            format: gpu::TextureFormat::Depth32Float,
            size: extent_i,
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
        });
        let depth_view_i = ctx.create_texture_view(
            depth_texture_i,
            gpu::TextureViewDesc {
                name: format!("depth view {i}").as_str(),
                format: gpu::TextureFormat::Depth32Float,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );
        let depth_sampler_i = ctx.create_sampler(gpu::SamplerDesc {
            name: format!("depth sampler {i}").as_str(),
            // compare: Some(gpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let depth_stuff_i = TextureStuff {
            texture: depth_texture_i,
            view: depth_view_i,
            sampler: depth_sampler_i,
            size: extent_i,
        };

        let pos_texture_i = ctx.create_texture(gpu::TextureDesc {
            name: format!("pos texture {i}").as_str(),
            format: gpu::TextureFormat::Rgba32Float,
            size: extent_i,
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
        });
        let pos_view_i = ctx.create_texture_view(
            pos_texture_i,
            gpu::TextureViewDesc {
                name: format!("pos view {i}").as_str(),
                format: gpu::TextureFormat::Rgba32Float,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );
        let pos_sampler_i = ctx.create_sampler(gpu::SamplerDesc {
            name: format!("pos sampler {i}").as_str(),
            address_modes: Default::default(),
            mag_filter: gpu::FilterMode::Nearest,
            min_filter: gpu::FilterMode::Nearest,
            mipmap_filter: gpu::FilterMode::Nearest,
            ..Default::default()
        });

        let pos_stuff_i = TextureStuff {
            texture: pos_texture_i,
            view: pos_view_i,
            sampler: pos_sampler_i,
            size: extent_i,
        };

        let normal_texture_i = ctx.create_texture(gpu::TextureDesc {
            name: format!("normal texture {i}").as_str(),
            format: gpu::TextureFormat::Rgba32Float,
            size: extent_i,
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
        });
        let normal_view_i = ctx.create_texture_view(
            normal_texture_i,
            gpu::TextureViewDesc {
                name: format!("normal view {i}").as_str(),
                format: gpu::TextureFormat::Rgba32Float,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );
        let normal_sampler_i = ctx.create_sampler(gpu::SamplerDesc {
            name: format!("normal sampler {i}").as_str(),
            address_modes: Default::default(),
            mag_filter: gpu::FilterMode::Nearest,
            min_filter: gpu::FilterMode::Nearest,
            mipmap_filter: gpu::FilterMode::Nearest,
            ..Default::default()
        });

        let normal_stuff_i = TextureStuff {
            texture: normal_texture_i,
            view: normal_view_i,
            sampler: normal_sampler_i,
            size: extent_i,
        };

        let depth_pos_normal_i = DepthPosNormalTexture {
            depth: depth_stuff_i,
            pos: pos_stuff_i,
            normal: normal_stuff_i,
        };

        depth_pos_normal_textures.push(depth_pos_normal_i);

        //NOTE: ao texture
        let ao_texture_i = ctx.create_texture(gpu::TextureDesc {
            name: format!("ao texture {i}").as_str(),
            format: gpu::TextureFormat::Rgba32Float,
            size: extent_i,
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
        });
        let ao_view_i = ctx.create_texture_view(
            ao_texture_i,
            gpu::TextureViewDesc {
                name: format!("ao view {i}").as_str(),
                format: gpu::TextureFormat::Rgba32Float,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );
        let ao_sampler_i = ctx.create_sampler(gpu::SamplerDesc {
            name: format!("ao sampler {i}").as_str(),
            address_modes: Default::default(),
            mag_filter: gpu::FilterMode::Nearest,
            min_filter: gpu::FilterMode::Nearest,
            mipmap_filter: gpu::FilterMode::Nearest,
            ..Default::default()
        });

        let ao_texture_stuff_i = TextureStuff {
            texture: ao_texture_i,
            view: ao_view_i,
            sampler: ao_sampler_i,
            size: extent_i,
        };

        ao_textures.push(ao_texture_stuff_i);

        //NOTE: ao texture after blur
        let ao_blur_texture_i = ctx.create_texture(gpu::TextureDesc {
            name: format!("ao blur texture {i}").as_str(),
            format: gpu::TextureFormat::Rgba32Float,
            size: extent_i,
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
        });
        let ao_blur_view_i = ctx.create_texture_view(
            ao_blur_texture_i,
            gpu::TextureViewDesc {
                name: format!("ao blur view {i}").as_str(),
                format: gpu::TextureFormat::Rgba32Float,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );
        let ao_blur_sampler_i = ctx.create_sampler(gpu::SamplerDesc {
            name: format!("ao blur sampler {i}").as_str(),
            address_modes: Default::default(),
            mag_filter: gpu::FilterMode::Nearest,
            min_filter: gpu::FilterMode::Nearest,
            mipmap_filter: gpu::FilterMode::Nearest,
            ..Default::default()
        });

        let ao_blur_texture_stuff_i = TextureStuff {
            texture: ao_blur_texture_i,
            view: ao_blur_view_i,
            sampler: ao_blur_sampler_i,
            size: extent_i,
        };

        ao_textures_blur.push(ao_blur_texture_stuff_i);
    }

    let ao_dummy_texture = {
        let dummy_extent = gpu::Extent {
            width: 1,
            height: 1,
            depth: 1,
        };
        let ao_texture_dummy = ctx.create_texture(gpu::TextureDesc {
            name: format!("ao texture dummy").as_str(),
            format: gpu::TextureFormat::Rgba32Float,
            size: dummy_extent,
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
        });
        let ao_view_dummy = ctx.create_texture_view(
            ao_texture_dummy,
            gpu::TextureViewDesc {
                name: format!("ao view dummy").as_str(),
                format: gpu::TextureFormat::Rgba32Float,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );
        let ao_sampler_dummy = ctx.create_sampler(gpu::SamplerDesc {
            name: format!("ao sampler dummy").as_str(),
            address_modes: Default::default(),
            mag_filter: gpu::FilterMode::Nearest,
            min_filter: gpu::FilterMode::Nearest,
            mipmap_filter: gpu::FilterMode::Nearest,
            ..Default::default()
        });
        TextureStuff {
            texture: ao_texture_dummy,
            view: ao_view_dummy,
            sampler: ao_sampler_dummy,
            size: dummy_extent,
        }
    };

    let downsample_textures = DownsampleTextures {
        textures: depth_pos_normal_textures,
    };
    let ao_textures = AOTextures {
        textures: ao_textures,
        dummy_texture: ao_dummy_texture,
        textures_after_blur: ao_textures_blur,
    };

    (downsample_textures, ao_textures)
}

pub struct Pipelines {
    // pub shader_paths: Vec<std::path::Path>,
    pub last_modified_shader_time: std::time::SystemTime,
    pub geometry: gpu::RenderPipeline,
    pub light: gpu::RenderPipeline,
    pub depth_downsample: gpu::RenderPipeline,
    pub calc_ao: gpu::RenderPipeline,
    pub blur_ao: gpu::RenderPipeline,
}

pub fn last_time_shader_modified() -> std::time::SystemTime {
    let geometry_shader_path = std::path::Path::new("src/shader.wgsl");
    let light_shader_path = std::path::Path::new("src/light_shader.wgsl");

    let mut t = std::time::SystemTime::UNIX_EPOCH;
    if let Ok(t1) = geometry_shader_path.metadata() {
        t = t.max(t1.modified().unwrap());
    }
    if let Ok(t2) = light_shader_path.metadata() {
        t = t.max(t2.modified().unwrap());
    }

    t
}

impl Pipelines {
    pub fn create_pipelines(ctx: &gpu::Context, surface: &gpu::Surface) -> Option<Self> {
        let geometry_shader_source = match std::fs::read_to_string("src/shader.wgsl") {
            Ok(src) => src,
            Err(err) => {
                dbg!(err);
                return None;
            }
        };

        let geometry_shader = match ctx.try_create_shader(gpu::ShaderDesc {
            source: &geometry_shader_source,
        }) {
            Ok(shader) => shader,
            Err(err) => {
                dbg!(err);
                return None;
            }
        };

        // NOTE: pipeline
        let geometry_pipeline = ctx.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "geometry",
            data_layouts: &[&<GeometryParams as gpu::ShaderData>::layout()],
            vertex: geometry_shader.at("vs_main"),
            vertex_fetches: &[gpu::VertexFetchState {
                layout: &<Vertex as gpu::Vertex>::layout(),
                instanced: false,
            }],
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleList,
                front_face: gpu::FrontFace::Ccw,
                cull_mode: Some(gpu::Face::Back),
                unclipped_depth: false,
                wireframe: false,
            },
            depth_stencil: Some(gpu::DepthStencilState {
                format: gpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: gpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: gpu::DepthBiasState::default(),
            }),
            fragment: geometry_shader.at("fs_main"),
            color_targets: &[
                gpu::ColorTargetState {
                    format: gpu::TextureFormat::Rgba32Float,
                    blend: Some(gpu::BlendState::REPLACE),
                    write_mask: gpu::ColorWrites::default(),
                },
                gpu::ColorTargetState {
                    format: gpu::TextureFormat::Rgba32Float,
                    blend: Some(gpu::BlendState::REPLACE),
                    write_mask: gpu::ColorWrites::default(),
                },
            ],
        });

        let light_shader_source = match std::fs::read_to_string("src/light_shader.wgsl") {
            Ok(src) => src,
            Err(err) => {
                dbg!(err);
                return None;
            }
        };
        let light_shader = match ctx.try_create_shader(gpu::ShaderDesc {
            source: &light_shader_source,
        }) {
            Ok(shader) => shader,
            Err(err) => {
                dbg!(err);
                return None;
            }
        };

        let light_pipeline = ctx.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "light",
            // data_layouts: &[&<Params as gpu::ShaderData>::layout()],
            data_layouts: &[&<LightPassParams as gpu::ShaderData>::layout()],
            vertex: light_shader.at("vs_main"),
            vertex_fetches: &[gpu::VertexFetchState {
                layout: &<Vertex as gpu::Vertex>::layout(),
                instanced: false,
            }],
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleList,
                front_face: gpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                wireframe: false,
            },
            depth_stencil: None,
            fragment: light_shader.at("fs_light"),
            color_targets: &[gpu::ColorTargetState {
                format: surface.info().format,
                blend: Some(gpu::BlendState::REPLACE),
                write_mask: gpu::ColorWrites::default(),
            }],
        });

        let depth_downsample_pipeline = ctx.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "depth downsample",
            data_layouts: &[&<DepthPosNormalParams as gpu::ShaderData>::layout()],
            vertex: light_shader.at("vs_main"),
            vertex_fetches: &[gpu::VertexFetchState {
                layout: &<Vertex as gpu::Vertex>::layout(),
                instanced: false,
            }],
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleList,
                front_face: gpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                wireframe: false,
            },
            depth_stencil: Some(gpu::DepthStencilState {
                format: gpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: gpu::CompareFunction::Always,
                stencil: gpu::StencilState::default(),
                bias: gpu::DepthBiasState::default(),
            }),
            fragment: light_shader.at("fs_downsample"),
            color_targets: &[
                gpu::ColorTargetState {
                    format: gpu::TextureFormat::Rgba32Float,
                    blend: Some(gpu::BlendState::REPLACE),
                    write_mask: gpu::ColorWrites::default(),
                },
                gpu::ColorTargetState {
                    format: gpu::TextureFormat::Rgba32Float,
                    blend: Some(gpu::BlendState::REPLACE),
                    write_mask: gpu::ColorWrites::default(),
                },
            ],
        });

        let ao_pipeline = ctx.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "ao",
            // TODO: fix daat layot
            data_layouts: &[&<CalcAoParams as gpu::ShaderData>::layout()],
            // data_layouts: &[],
            vertex: light_shader.at("vs_main"),
            vertex_fetches: &[gpu::VertexFetchState {
                layout: &<Vertex as gpu::Vertex>::layout(),
                instanced: false,
            }],
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleList,
                front_face: gpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                wireframe: false,
            },
            depth_stencil: None,
            fragment: light_shader.at("fs_calc_ao"),
            color_targets: &[gpu::ColorTargetState {
                format: gpu::TextureFormat::Rgba32Float,
                blend: Some(gpu::BlendState::REPLACE),
                write_mask: gpu::ColorWrites::default(),
            }],
        });

        let ao_blur = ctx.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "ao blur",
            data_layouts: &[&<BlurParams as gpu::ShaderData>::layout()],
            vertex: light_shader.at("vs_main"),
            vertex_fetches: &[gpu::VertexFetchState {
                layout: &<Vertex as gpu::Vertex>::layout(),
                instanced: false,
            }],
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleList,
                front_face: gpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                wireframe: false,
            },
            depth_stencil: None,
            fragment: light_shader.at("fs_blur_ao"),
            color_targets: &[gpu::ColorTargetState {
                format: gpu::TextureFormat::Rgba32Float,
                blend: Some(gpu::BlendState::REPLACE),
                write_mask: gpu::ColorWrites::default(),
            }],
        });

        let last_modified = last_time_shader_modified();
        // let metadata = std::fs::Metadata:
        Some(Self {
            geometry: geometry_pipeline,
            light: light_pipeline,
            depth_downsample: depth_downsample_pipeline,
            last_modified_shader_time: last_modified,
            calc_ao: ao_pipeline,
            blur_ao: ao_blur,
        })
    }
}

pub struct State {
    pub delta_time: f32,
    pub prev_time: std::time::SystemTime,
    pub pipelines: Pipelines,
    pub command_encoder: gpu::CommandEncoder,
    pub ctx: gpu::Context,
    pub surface: gpu::Surface,
    pub prev_sync_point: Option<gpu::SyncPoint>,
    pub meshes: Vec<Mesh>,
    pub camera: Camera,
    pub retained_input: RetainedInput,
    pub screen_quad_buf: gpu::BufferPiece,
    pub downsample_textures: DownsampleTextures,
    pub ao_textures: AOTextures,
    pub input_state: InputState,
}

#[derive(Default)]
pub struct RetainedInput {
    pub just_pressed_keys: std::collections::HashSet<winit::keyboard::KeyCode>,
    pub held_keys: std::collections::HashSet<winit::keyboard::KeyCode>,
}

impl State {
    pub fn new(window: &winit::window::Window) -> Self {
        let ctx = unsafe {
            gpu::Context::init(gpu::ContextDesc {
                presentation: true,
                validation: true,
                timing: false,
                capture: false,
                overlay: false,
                device_id: 0,
            })
            .unwrap()
        };
        let size = window.inner_size();
        let width = size.width;
        let height = size.height;

        let screen_extent = gpu::Extent {
            width,
            height,
            depth: 1,
        };

        let aspect = width as f32 / height as f32;
        let surface = ctx
            .create_surface_configured(
                window,
                gpu::SurfaceConfig {
                    size: screen_extent,
                    usage: gpu::TextureUsage::TARGET,
                    display_sync: gpu::DisplaySync::Recent,
                    ..Default::default()
                },
            )
            .unwrap();

        let mut meshes = vec![];

        let command_encoder = ctx.create_command_encoder(gpu::CommandEncoderDesc {
            name: "main",
            buffer_count: 1,
        });

        let sponza_vertices = load_sponza();
        let sibekik_cathedrals = load_cathedral();
        let a = sponza_vertices.len() / 3;
        dbg!(a);
        // let gpu_sponza = upload_vertices(sponza_vertices, &ctx);
        let gpu_sponza = upload_vertices(sibekik_cathedrals, &ctx);
        meshes.clear();
        meshes.push(gpu_sponza);

        // let g_buffer = GBuffer::new(&ctx, width, height);

        let screen_size = gpu::Extent {
            width,
            height,
            depth: 1,
        };

        let (downsample_textures, ao_textures) =
            create_downsample_and_ao_textures(&ctx, screen_size);

        let screen_quad_vertices = [
            vec3(-1.0, -1.0, 0.0),
            vec3(1.0, -1.0, 0.0),
            vec3(-1.0, 1.0, 0.0),
            vec3(1.0, -1.0, 0.0),
            vec3(1.0, 1.0, 0.0),
            vec3(-1.0, 1.0, 0.0),
        ]
        .map(|a| Vertex {
            ws_pos: a.to_array(),
            ws_normal: Default::default(),
        });

        let screen_quad_buf = ctx.create_buffer(gpu::BufferDesc {
            name: "screen quad buf",
            size: (screen_quad_vertices.len() * std::mem::size_of::<Vertex>()) as u64,
            memory: gpu::Memory::Shared,
        });
        unsafe {
            std::ptr::copy_nonoverlapping(
                screen_quad_vertices.as_ptr(),
                screen_quad_buf.data() as *mut Vertex,
                screen_quad_vertices.len(),
            );
        }
        ctx.sync_buffer(screen_quad_buf);

        // let depth_textures = create_depth_textures(&ctx, screen_extent);

        let pipelines = Pipelines::create_pipelines(&ctx, &surface).unwrap();

        let input_state = InputState {
            ao_level: 0,
            use_blur: false,
        };

        Self {
            command_encoder,
            ctx,
            surface,
            prev_sync_point: None,
            meshes,
            camera: Camera::default_from_aspect(aspect),
            retained_input: Default::default(),
            screen_quad_buf: screen_quad_buf.into(),
            pipelines,
            downsample_textures,
            ao_textures,
            input_state,
            delta_time: 0.1,
            prev_time: std::time::SystemTime::now(),
        }
    }

    pub fn render_downsample(&mut self) {
        for i in 1..NUM_AO_TEXTURES {
            let textures_from = &self.downsample_textures.textures[i - 1];
            let textures_to = &self.downsample_textures.textures[i];

            if let mut depth_downsample_pass = self.command_encoder.render(
                format!("depth downsample {i}").as_str(),
                gpu::RenderTargetSet {
                    colors: &[
                        gpu::RenderTarget {
                            view: textures_to.pos.view,
                            init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                            finish_op: gpu::FinishOp::Store,
                        },
                        gpu::RenderTarget {
                            view: textures_to.normal.view,
                            init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                            finish_op: gpu::FinishOp::Store,
                        },
                    ],
                    depth_stencil: Some(gpu::RenderTarget {
                        view: textures_to.depth.view,
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                        finish_op: gpu::FinishOp::Store,
                    }),
                },
            ) {
                let mut rc = depth_downsample_pass.with(&self.pipelines.depth_downsample);
                rc.bind(
                    0,
                    &DepthPosNormalParams {
                        depth_view: textures_from.depth.view,
                        depth_sampler: textures_from.depth.sampler,
                        pos_view: textures_from.pos.view,
                        pos_sampler: textures_from.pos.sampler,
                        normal_view: textures_from.normal.view,
                        normal_sampler: textures_from.normal.sampler,
                        globals: Globals {
                            mvp_transform: self.camera.vp().to_cols_array_2d(),
                            mv_transform: self.camera.view().to_cols_array_2d(),
                            mv_rot: self.camera.view_rot_only().to_cols_array_2d(),
                            cam_pos: self.camera.pos.to_array(),
                            cam_dir: self.camera.right_forward_up()[1].to_array(),
                            pad: [0; 2],
                        },
                    },
                );
                rc.bind_vertex(0, self.screen_quad_buf);
                let num_quad_vertices = 6;
                rc.draw(0, num_quad_vertices as _, 0, 1);
            }
        }
    }

    pub fn render_calc_ao(&mut self) {
        for i in (0..NUM_AO_TEXTURES).rev() {
            let ao_target = &self.ao_textures.textures[i];
            let is_first_pass = i == NUM_AO_TEXTURES - 1;

            let ao_prev = if is_first_pass {
                &self.ao_textures.dummy_texture
            } else {
                &self.ao_textures.textures[i + 1]
            };

            // NOTE: calc ao pass
            if let mut calc_ao_pass = self.command_encoder.render(
                format!("calc ao {i}").as_str(),
                gpu::RenderTargetSet {
                    colors: &[gpu::RenderTarget {
                        view: ao_target.view,
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                        finish_op: gpu::FinishOp::Store,
                    }],
                    depth_stencil: None,
                },
            ) {
                // NOTE: these textures have same size as render target
                let dnp = &self.downsample_textures.textures[i];
                let prev_dnp = &self.downsample_textures.textures[(i + 1).min(NUM_AO_TEXTURES - 1)];
                let prev_ao_blur =
                    &self.ao_textures.textures_after_blur[(i + 1).min(NUM_AO_TEXTURES - 1)];
                let mut rc = calc_ao_pass.with(&self.pipelines.calc_ao);

                rc.bind(
                    0,
                    &CalcAoParams {
                        pos_view: dnp.pos.view,
                        pos_sampler: dnp.pos.sampler,

                        normal_view: dnp.normal.view,
                        normal_sampler: dnp.normal.sampler,

                        prev_pos_view: prev_dnp.pos.view,
                        prev_pos_sampler: prev_dnp.pos.sampler,

                        prev_normal_view: prev_dnp.normal.view,
                        prev_normal_sampler: prev_dnp.normal.sampler,

                        prev_ao_view: prev_ao_blur.view,
                        prev_ao_sampler: prev_ao_blur.sampler,

                        ao_params: AOParams::from(
                            i,
                            1.0,
                            self.camera.vfov_rad,
                            ao_target.size.width,
                            ao_target.size.height,
                        ),
                    },
                );
                rc.bind_vertex(0, self.screen_quad_buf);
                let num_quad_vertices = 6;
                rc.draw(0, num_quad_vertices as _, 0, 1);
            }

            // NOTE: blur ao pass

            let ao_blur_target = &self.ao_textures.textures_after_blur[i];
            if let mut blur_ao_pass = self.command_encoder.render(
                format!("blur ao {i}").as_str(),
                gpu::RenderTargetSet {
                    colors: &[gpu::RenderTarget {
                        view: ao_blur_target.view,
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                        finish_op: gpu::FinishOp::Store,
                    }],
                    depth_stencil: None,
                },
            ) {
                // NOTE: these textures have same size as render target
                let mut rc = blur_ao_pass.with(&self.pipelines.blur_ao);

                rc.bind(
                    0,
                    &BlurParams {
                        ao_view: ao_target.view,
                        ao_sampler: ao_target.sampler,
                        ao_params: AOParams::from(
                            i,
                            1.0,
                            self.camera.vfov_rad,
                            ao_target.size.width,
                            ao_target.size.height,
                        ),
                    },
                );
                rc.bind_vertex(0, self.screen_quad_buf);
                let num_quad_vertices = 6;
                rc.draw(0, num_quad_vertices as _, 0, 1);
            }
        }
    }

    pub fn render(&mut self) {
        self.command_encoder.start();
        for texture in self.downsample_textures.textures.iter() {
            self.command_encoder.init_texture(texture.depth.texture);
            self.command_encoder.init_texture(texture.pos.texture);
            self.command_encoder.init_texture(texture.normal.texture);
        }

        for t in self.ao_textures.textures.iter() {
            self.command_encoder.init_texture(t.texture);
        }
        for t in self.ao_textures.textures_after_blur.iter() {
            self.command_encoder.init_texture(t.texture);
        }
        self.command_encoder
            .init_texture(self.ao_textures.dummy_texture.texture);

        let geometry_target = &self.downsample_textures.textures[0];

        if let mut geometry_pass = self.command_encoder.render(
            "geometry",
            gpu::RenderTargetSet {
                colors: &[
                    gpu::RenderTarget {
                        view: geometry_target.pos.view,
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                        finish_op: gpu::FinishOp::Store,
                    },
                    gpu::RenderTarget {
                        view: geometry_target.normal.view,
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                        finish_op: gpu::FinishOp::Store,
                    },
                ],
                depth_stencil: Some(gpu::RenderTarget {
                    view: geometry_target.depth.view,
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                    finish_op: gpu::FinishOp::Store,
                }),
            },
        ) {
            let mut rc = geometry_pass.with(&self.pipelines.geometry);
            rc.bind(
                0,
                &GeometryParams {
                    globals: Globals {
                        mvp_transform: self.camera.vp().to_cols_array_2d(),
                        mv_transform: self.camera.view().to_cols_array_2d(),
                        cam_pos: self.camera.pos.to_array(),
                        cam_dir: self.camera.right_forward_up()[1].to_array(),
                        pad: [0; 2],
                        mv_rot: self.camera.view_rot_only().to_cols_array_2d(),
                    },
                },
            );

            for mesh in self.meshes.iter() {
                rc.bind_vertex(0, mesh.vertex_buf);
                rc.draw(0, mesh.num_vertices as _, 0, 1);
            }
        }

        self.render_downsample();
        self.render_calc_ao();

        let textures_for_light_pass = &self.downsample_textures.textures[0];
        // let textures_for_light_pass = &self.downsample_textures.textures.last().unwrap();
        let frame = self.surface.acquire_frame();
        self.command_encoder.init_texture(frame.texture());
        if let mut light_pass = self.command_encoder.render(
            "light",
            gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: frame.texture_view(),
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: None,
            },
        ) {
            let mut rc = light_pass.with(&self.pipelines.light);

            let use_blurred_texture = self.input_state.use_blur;
            let ao_index = self.input_state.ao_level;
            let ao_texture = if use_blurred_texture {
                &self.ao_textures.textures_after_blur[ao_index]
            } else {
                &self.ao_textures.textures[ao_index]
            };
            // let ao_texture = &self.ao_textures.textures[0];
            // let ao_texture = &self.downsample_textures.textures[1].normal;
            rc.bind(
                0,
                &LightPassParams {
                    pos_view: textures_for_light_pass.pos.view,
                    pos_sampler: textures_for_light_pass.pos.sampler,
                    normal_view: textures_for_light_pass.normal.view,
                    normal_sampler: textures_for_light_pass.normal.sampler,
                    depth_view: textures_for_light_pass.depth.view,
                    depth_sampler: textures_for_light_pass.depth.sampler,
                    globals: Globals {
                        mvp_transform: self.camera.vp().to_cols_array_2d(),
                        mv_transform: self.camera.view().to_cols_array_2d(),
                        cam_pos: self.camera.pos.to_array(),
                        cam_dir: self.camera.right_forward_up()[1].to_array(),
                        pad: [0; 2],
                        mv_rot: self.camera.view_rot_only().to_cols_array_2d(),
                    },
                    ao_view: ao_texture.view,
                    ao_sampler: ao_texture.sampler,
                },
            );
            rc.bind_vertex(0, self.screen_quad_buf);
            let num_quad_vertices = 6;
            rc.draw(0, num_quad_vertices as _, 0, 1);
        }
        self.command_encoder.present(frame);

        let sp = self.ctx.submit(&mut self.command_encoder);
        self.ctx.wait_for(&sp, !0);
    }

    pub fn handle_input(&mut self) {
        let [r, f, u] = self.camera.right_forward_up();

        let speed = 6.0;
        let angle_speed = 0.8;
        let dt = self.delta_time;

        for key in self.retained_input.held_keys.iter() {
            match key {
                winit::keyboard::KeyCode::KeyW => {
                    self.camera.pos += f * dt * speed;
                }
                winit::keyboard::KeyCode::KeyA => {
                    self.camera.pos -= r * dt * speed;
                }
                winit::keyboard::KeyCode::KeyS => {
                    self.camera.pos -= f * dt * speed;
                }
                winit::keyboard::KeyCode::KeyD => {
                    self.camera.pos += r * dt * speed;
                }
                winit::keyboard::KeyCode::KeyQ => {
                    self.camera.pos -= u * dt * speed;
                }
                winit::keyboard::KeyCode::KeyE => {
                    self.camera.pos += u * dt * speed;
                }

                // angle
                winit::keyboard::KeyCode::KeyI => {
                    self.camera.pitch += dt * angle_speed;
                }
                winit::keyboard::KeyCode::KeyJ => {
                    self.camera.yaw += dt * angle_speed;
                }
                winit::keyboard::KeyCode::KeyK => {
                    self.camera.pitch -= dt * angle_speed;
                }
                winit::keyboard::KeyCode::KeyL => {
                    self.camera.yaw -= dt * angle_speed;
                }

                winit::keyboard::KeyCode::Digit1 => {
                    self.input_state.ao_level = 0;
                }
                winit::keyboard::KeyCode::Digit2 => {
                    self.input_state.ao_level = 1;
                }
                winit::keyboard::KeyCode::Digit3 => {
                    self.input_state.ao_level = 2;
                }
                winit::keyboard::KeyCode::Digit4 => {
                    self.input_state.ao_level = 3;
                }
                winit::keyboard::KeyCode::Digit5 => {
                    self.input_state.ao_level = 4;
                }

                winit::keyboard::KeyCode::Digit6 => {
                    self.input_state.ao_level = 5;
                }
                winit::keyboard::KeyCode::Digit7 => {
                    self.input_state.ao_level = 6;
                }

                _ => {}
            }

            self.input_state.ao_level = self.input_state.ao_level.min(NUM_AO_TEXTURES - 1);

            for key in self.retained_input.just_pressed_keys.iter() {
                match key {
                    winit::keyboard::KeyCode::KeyB => {
                        self.input_state.use_blur = !self.input_state.use_blur;

                        dbg!(self.input_state.use_blur);
                    }
                    winit::keyboard::KeyCode::KeyU => {
                        self.camera.save_state();
                    }
                    winit::keyboard::KeyCode::KeyY => {
                        self.camera.load_state();
                    }

                    _ => {}
                }
            }
        }
        self.retained_input.just_pressed_keys.clear();
    }

    pub fn recreate_pipelines_if_required(&mut self) {
        // let geometry_shader_source = std::fs::read_to_string().unwrap();
        let shader_modified_time = last_time_shader_modified();
        if self.pipelines.last_modified_shader_time != shader_modified_time {
            self.pipelines.last_modified_shader_time = shader_modified_time;

            if let Some(new_pipelines) = Pipelines::create_pipelines(&self.ctx, &self.surface) {
                dbg!("recompiled all shaders");
                self.pipelines = new_pipelines;
            }
        }
    }
}

impl Camera {
    pub fn view(&self) -> glam::Mat4 {
        let rot = self.rot_quat();
        let pos = Vec3::from_array(self.pos.to_array());
        let view = Mat4::from_scale_rotation_translation(Vec3A::ONE.into(), rot, pos).inverse();
        view
    }

    pub fn view_rot_only(&self) -> glam::Mat4 {
        let rot = self.rot_quat();
        let mat = glam::Mat4::from_rotation_translation(rot, Vec3::ZERO).inverse();
        mat
    }

    pub fn rot_quat(&self) -> glam::Quat {
        let rot_x = Quat::from_axis_angle(Vec3::X, self.pitch);
        let rot_y = Quat::from_axis_angle(Vec3::Y, self.yaw);
        let rot = rot_y * rot_x;
        rot
    }

    pub fn projection(&self) -> glam::Mat4 {
        glam::Mat4::perspective_rh(self.vfov_rad, self.aspect, 0.001, 100.0)
    }

    pub fn default_from_aspect(aspect: f32) -> Self {
        Self {
            pos: Vec3A::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            // vfov_rad: TAU / 4.0,
            vfov_rad: 70.0_f32.to_radians(),
            aspect,
        }
    }

    pub fn vp(&self) -> glam::Mat4 {
        let v = self.view();
        let p = self.projection();
        // dbg!(v);
        p * v
    }

    pub fn right_forward_up(&self) -> [Vec3A; 3] {
        let v = self.view();
        let rot = v.to_scale_rotation_translation().1.inverse();

        let r = rot * Vec3A::X;
        let f = rot * -Vec3A::Z;
        let u = rot * Vec3A::Y;

        [r, f, u]
    }
    pub fn save_state(&self) {
        let path = std::path::Path::new("src/assets/cam/cam.txt");
        let Ok(file) = std::fs::File::create(path) else {
            dbg!("coulf not write cam state");
            return;
        };
        let mut w = std::io::BufWriter::new(file);

        let _ = w.write_fmt(format_args!(" {}", self.pos.x));
        let _ = w.write_fmt(format_args!(" {}", self.pos.y));
        let _ = w.write_fmt(format_args!(" {}", self.pos.z));

        let _ = w.write_fmt(format_args!(" {}", self.yaw));
        let _ = w.write_fmt(format_args!(" {}", self.pitch));
        let _ = w.write_fmt(format_args!(" {}", self.vfov_rad));
        let _ = w.write_fmt(format_args!(" {}", self.aspect));
        dbg!("saved cam state to file");
        // self.
    }
    pub fn load_state(&mut self) {
        let path = std::path::Path::new("src/assets/cam/cam.txt");
        let Ok(file) = std::fs::File::open(path) else {
            dbg!("coulf not read cam state");
            return;
        };
        let reader = std::io::BufReader::new(file);
        let Some(Ok(lines)) = reader.lines().next() else {
            dbg!("coulf not load cam state");
            return;
        };
        let mut args = lines.split_whitespace();

        self.pos.x = args.next().unwrap().parse().unwrap();
        self.pos.y = args.next().unwrap().parse().unwrap();
        self.pos.z = args.next().unwrap().parse().unwrap();

        self.yaw = args.next().unwrap().parse().unwrap();
        self.pitch = args.next().unwrap().parse().unwrap();
        self.vfov_rad = args.next().unwrap().parse().unwrap();
        self.aspect = args.next().unwrap().parse().unwrap();
    }
}
pub fn load_sponza() -> Vec<Vertex> {
    dbg!("loading sponza");
    let path = std::path::Path::new("src/assets/sponza/sponza.obj");
    let mesh = parse_obj_file(path);
    let vertices = turn_mesh_into_pure_vertex_list(mesh);

    vertices
}

pub fn load_cathedral() -> Vec<Vertex> {
    dbg!("loading sibenik cathedral");
    let path = std::path::Path::new("src/assets/sibenik_cathedral/sibenik.obj");
    let mesh = parse_obj_file(path);
    let vertices = turn_mesh_into_pure_vertex_list(mesh);

    vertices
}

// pub fn load_

pub fn turn_mesh_into_pure_vertex_list(mesh: CpuMesh) -> Vec<Vertex> {
    let mut vertices = vec![];

    for idxs in mesh.indices.chunks_exact(3) {
        let i0 = idxs[0];
        let i1 = idxs[1];
        let i2 = idxs[2];

        let v0 = mesh.vertices[i0];
        let v1 = mesh.vertices[i1];
        let v2 = mesh.vertices[i2];
        let n = (v1 - v0).cross(v2 - v0).normalize();

        for pos in [v0, v1, v2] {
            let new_vertex = Vertex {
                ws_pos: pos.to_array(),
                ws_normal: n.to_array(),
            };
            vertices.push(new_vertex);
        }
    }

    vertices
}

pub fn upload_vertices(vertices: Vec<Vertex>, ctx: &gpu::Context) -> Mesh {
    let vertex_buf = ctx.create_buffer(gpu::BufferDesc {
        name: "vertex buffer",
        size: (vertices.len() * std::mem::size_of::<Vertex>()) as u64,
        memory: gpu::Memory::Shared,
    });
    unsafe {
        std::ptr::copy_nonoverlapping(
            vertices.as_ptr(),
            vertex_buf.data() as *mut Vertex,
            vertices.len(),
        );
    }
    let mesh = Mesh {
        vertex_buf: vertex_buf.into(),
        index_buf: None,
        num_vertices: vertices.len(),
        num_indices: 0,
    };

    ctx.sync_buffer(vertex_buf);
    mesh
}

pub fn upload_mesh(ctx: &gpu::Context, mesh: CpuMesh) -> Mesh {
    let CpuMesh { vertices, indices } = mesh;

    let normals = indices
        .chunks(3)
        .map(|idxs| {
            let i0 = idxs[0];
            let i1 = idxs[1];
            let i2 = idxs[2];

            let v0 = vertices[i0];
            let v1 = vertices[i1];
            let v2 = vertices[i2];
            let n = (v1 - v0).cross(v2 - v0).normalize();
            n
        })
        .collect::<Vec<_>>();
    let gpu_vertices = vertices
        .iter()
        .enumerate()
        .map(|(i, v)| Vertex {
            ws_pos: v.to_array(),
            ws_normal: normals[i / 3].to_array(),
        })
        .collect::<Vec<_>>();
    let vertex_buf = ctx.create_buffer(gpu::BufferDesc {
        name: "vertex buffer",
        size: (vertices.len() * std::mem::size_of::<Vertex>()) as u64,
        memory: gpu::Memory::Shared,
    });
    unsafe {
        std::ptr::copy_nonoverlapping(
            gpu_vertices.as_ptr(),
            vertex_buf.data() as *mut Vertex,
            vertices.len(),
        );
    }
    let indices = indices.iter().map(|idx| *idx as u32).collect::<Vec<_>>();
    let index_buf = ctx.create_buffer(gpu::BufferDesc {
        name: "index buffer",
        size: (indices.len() * std::mem::size_of::<u32>()) as u64,
        memory: gpu::Memory::Shared,
    });

    unsafe {
        std::ptr::copy_nonoverlapping(
            indices.as_ptr(),
            index_buf.data() as *mut u32,
            indices.len(),
        );
    }

    let mesh = Mesh {
        vertex_buf: vertex_buf.into(),
        index_buf: Some(index_buf.into()),
        num_vertices: vertices.len(),
        num_indices: indices.len(),
    };

    ctx.sync_buffer(vertex_buf);
    ctx.sync_buffer(index_buf);

    mesh
}

pub fn parse_obj_file<P: AsRef<std::path::Path>>(path: P) -> CpuMesh {
    let mut vertices = vec![];
    let mut normals = vec![];
    let mut indices = vec![];
    // pub fn parse_obj_file<R: std::io::BufRead>(file: R) {
    if let Ok(file) = std::fs::File::open(path) {
        let mut reader = std::io::BufReader::new(file);
        let mut lines = reader.lines();
        while let Some(Ok(line)) = lines.next() {
            if let Some((pre, rest)) = line.split_once(" ") {
                match pre {
                    "v" => {
                        let mut v = Vec3A::ZERO;
                        for (i, x) in rest.split_whitespace().enumerate() {
                            if let Ok(x) = x.parse() {
                                if i > 2 {
                                    dbg!(&line);
                                }
                                v[i] = x;
                            }
                        }
                        vertices.push(v);
                    }
                    "vn" => {
                        let mut v = Vec3A::ZERO;
                        for (i, x) in rest.split(" ").enumerate() {
                            if let Ok(x) = x.parse() {
                                v[i] = x;
                            }
                        }
                        normals.push(v);
                    }
                    "f" => {
                        let vals = rest.split(" ");
                        let mut these_indices = vec![];
                        for val in vals {
                            if let Some((v_idx, uv_idx)) = val.split_once("/") {
                                if let Ok(v_idx) = v_idx.parse::<usize>() {
                                    // NOTE: obj uses 1-based indices
                                    these_indices.push(v_idx - 1);
                                }
                            }
                        }
                        let n = these_indices.len();
                        match n {
                            3 => {
                                indices.extend(these_indices);
                            }
                            4 => {
                                indices.push(these_indices[0]);
                                indices.push(these_indices[1]);
                                indices.push(these_indices[2]);

                                indices.push(these_indices[2]);
                                indices.push(these_indices[3]);
                                indices.push(these_indices[0]);
                            }
                            _ => {
                                dbg!(format!("weird idx len {n}"));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        // for line in reader.lines() {
        //     let (a, rest)
        //     if let Some
        //     // dbg!(line);
        // }
        // while let Some(line) = file.read_line()
    }

    dbg!(vertices.len());
    dbg!(normals.len());
    dbg!(indices.len());

    CpuMesh { vertices, indices }
}

fn main() {
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let window_attributes = winit::window::Window::default_attributes()
        .with_title("ssao")
        .with_inner_size(winit::dpi::PhysicalSize::new(1024, 1024))
        // .with_inner_size(winit::dpi::PhysicalSize::new(512, 512))
        // .with_inner_size(winit::dpi::PhysicalSize::new(2048,2048))
        // .with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
        ;

    let window = event_loop.create_window(window_attributes).unwrap();

    let mut state = State::new(&window);

    event_loop
        .run(|event, target| {
            target.set_control_flow(winit::event_loop::ControlFlow::Poll);
            match event {
                winit::event::Event::AboutToWait => window.request_redraw(),
                winit::event::Event::WindowEvent { event, .. } => match event {
                    winit::event::WindowEvent::Resized(_) => {}
                    winit::event::WindowEvent::KeyboardInput {
                        event:
                            winit::event::KeyEvent {
                                physical_key: winit::keyboard::PhysicalKey::Code(key_code),
                                state: key_state,
                                ..
                            },
                        ..
                    } => match key_state {
                        winit::event::ElementState::Pressed => {
                            if state.retained_input.held_keys.insert(key_code) {
                                state.retained_input.just_pressed_keys.insert(key_code);
                            }
                        }
                        winit::event::ElementState::Released => {
                            state.retained_input.held_keys.remove(&key_code);
                        }
                    },
                    winit::event::WindowEvent::CloseRequested => {
                        dbg!("closing");
                        target.exit();
                    }
                    winit::event::WindowEvent::RedrawRequested => {
                        let now = std::time::SystemTime::now();
                        if let Ok(delta) = now.duration_since(state.prev_time) {
                            state.delta_time = delta.as_secs_f32();
                        }
                        state.prev_time = now;
                        state.recreate_pipelines_if_required();
                        state.handle_input();
                        state.render();
                    }
                    _ => {}
                },
                _ => {}
            }
        })
        .unwrap();
}
