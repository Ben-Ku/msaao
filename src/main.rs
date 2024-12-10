use std::io::BufRead;

use nanorand::Rng;

pub use blade_graphics as gpu;
use bytemuck::{Pod, Zeroable};
pub use glam::*;

pub const PI: f32 = 3.14159265358979323846264338327950288;
pub const TAU: f32 = 2.0 * PI;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Globals {
    mvp_transform: [[f32; 4]; 4],
    cam_pos: [f32; 3],
    cam_dir: [f32; 3],
    pad: [u32; 2],
}

// pub LightGlobals {

// }

#[derive(blade_macros::ShaderData)]
pub struct Params {
    pub globals: Globals,
    pub depth_view: gpu::TextureView,
    pub depth_sampler: gpu::Sampler,
}

#[derive(blade_macros::ShaderData)]
pub struct LightParams {
    pub pos_view: gpu::TextureView,
    pub pos_sampler: gpu::Sampler,

    pub normal_view: gpu::TextureView,
    pub normal_sampler: gpu::Sampler,

    pub depth_view: gpu::TextureView,
    pub depth_sampler: gpu::Sampler,
    // pub near: f32,
    // pub far: f32,
    // pub pos_view: gpu::TextureView,
    // pub pos_sampler: gpu::Sampler,
}

#[derive(blade_macros::ShaderData)]
pub struct DepthDownsampleParams {
    pub depth_from: gpu::TextureView,
    pub depth_from_sampler: gpu::Sampler,
}

#[derive(blade_macros::Vertex, Debug)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
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

pub struct Camera {
    pub pos: Vec3A,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_rad: f32,
    pub aspect: f32,
}

pub struct GBuffer {
    // pub geometry_shader: gpu::Shader,
    // pub geometry_pipeline: gpu::RenderPipeline,
    pub depth_textures: DepthTextures,
    pub pos_texture: gpu::Texture,
    pub normal_texture: gpu::Texture,

    // pub depth_view: gpu::TextureView,
    pub pos_view: gpu::TextureView,
    pub normal_view: gpu::TextureView,

    // pub depth_sampler: gpu::Sampler,
    pub pos_sampler: gpu::Sampler,
    pub normal_sampler: gpu::Sampler,
}

pub struct TextureStuff {
    pub texture: gpu::Texture,
    pub view: gpu::TextureView,
    pub sampler: gpu::Sampler,
    pub size: gpu::Extent,
}

pub struct DepthTextures {
    pub texture_stuffs: Vec<TextureStuff>,
}

pub fn create_depth_textures(ctx: &gpu::Context, screen_size: gpu::Extent) -> DepthTextures {
    let mut texture_stuffs = vec![];
    let num_textures = 5;
    let mut width = screen_size.width;
    let mut height = screen_size.height;

    for i in 0..num_textures {
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

        let texture_stuff_i = TextureStuff {
            texture: depth_texture_i,
            view: depth_view_i,
            sampler: depth_sampler_i,
            size: extent_i,
        };
        texture_stuffs.push(texture_stuff_i);
    }

    DepthTextures { texture_stuffs }
}

pub struct Pipelines {
    // pub shader_paths: Vec<std::path::Path>,
    pub last_modified_shader_time: std::time::SystemTime,
    pub geometry: gpu::RenderPipeline,
    pub light: gpu::RenderPipeline,
    pub depth_downsample: gpu::RenderPipeline,
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
            data_layouts: &[&<Params as gpu::ShaderData>::layout()],
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
            data_layouts: &[&<LightParams as gpu::ShaderData>::layout()],
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
            fragment: light_shader.at("fs_main"),
            color_targets: &[gpu::ColorTargetState {
                format: surface.info().format,
                blend: Some(gpu::BlendState::REPLACE),
                write_mask: gpu::ColorWrites::default(),
            }],
        });

        let depth_downsample_pipeline = ctx.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "depth downsample",
            data_layouts: &[&<DepthDownsampleParams as gpu::ShaderData>::layout()],
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
                depth_write_enabled: false,
                depth_compare: gpu::CompareFunction::Never,
                stencil: gpu::StencilState::default(),
                bias: gpu::DepthBiasState::default(),
            }),
            fragment: light_shader.at("fs_downsample"),
            color_targets: &[],
        });

        let last_modified = last_time_shader_modified();
        // let metadata = std::fs::Metadata:
        Some(Self {
            geometry: geometry_pipeline,
            light: light_pipeline,
            depth_downsample: depth_downsample_pipeline,
            last_modified_shader_time: last_modified,
        })
    }

    // pub fn new(ctx: &gpu::Context) {
    //     // let fill_gbuffer =  ctx.create_render_pipeline(
    //     //     gpu::RenderPipelineDesc {
    //     //      name: "fill gbuffer",
    //     //      data_layouts: todo!(),
    //     //      vertex: todo!(),
    //     //      vertex_fetches: todo!(),
    //     //      primitive: todo!(),
    //     //      depth_stencil: todo!(),
    //     //      fragment: todo!(),
    //     //      color_targets: todo!(),
    //     //  }
    //     // );
    //     let pipeline = ctx.create_render_pipeline(gpu::RenderPipelineDesc {
    //         name: "geometry",
    //         data_layouts: &[&<Params as gpu::ShaderData>::layout()],
    //         vertex: shader.at("vs_main"),
    //         vertex_fetches: &[gpu::VertexFetchState {
    //             layout: &<Vertex as gpu::Vertex>::layout(),
    //             instanced: false,
    //         }],
    //         primitive: gpu::PrimitiveState {
    //             topology: gpu::PrimitiveTopology::TriangleList,
    //             front_face: gpu::FrontFace::Ccw,
    //             cull_mode: Some(gpu::Face::Back),
    //             unclipped_depth: false,
    //             wireframe: false,
    //         },
    //         depth_stencil: Some(gpu::DepthStencilState {
    //             format: gpu::TextureFormat::Depth32Float,
    //             depth_write_enabled: true,
    //             depth_compare: gpu::CompareFunction::Less,
    //             stencil: Default::default(),
    //             bias: gpu::DepthBiasState::default(),
    //         }),
    //         fragment: shader.at("fs_main"),
    //         color_targets: &[gpu::ColorTargetState {
    //             format: surface.info().format,
    //             blend: Some(gpu::BlendState::ALPHA_BLENDING),
    //             write_mask: gpu::ColorWrites::default(),
    //         }],
    //     });
    // }
}

impl GBuffer {
    pub fn new(ctx: &gpu::Context, width: u32, height: u32) -> Self {
        let extent = gpu::Extent {
            width,
            height,
            depth: 1,
        };

        dbg!(extent);
        let depth_texture = ctx.create_texture(gpu::TextureDesc {
            name: "depth texture",
            format: gpu::TextureFormat::Depth32Float,
            size: extent,
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
        });
        let depth_view = ctx.create_texture_view(
            depth_texture,
            gpu::TextureViewDesc {
                name: "depth view",
                format: gpu::TextureFormat::Depth32Float,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );
        let depth_sampler = ctx.create_sampler(gpu::SamplerDesc {
            name: "depth sampler",
            // compare: Some(gpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let pos_texture = ctx.create_texture(gpu::TextureDesc {
            name: "pos texture",
            format: gpu::TextureFormat::Rgba32Float,
            size: extent,
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
        });
        let pos_view = ctx.create_texture_view(
            pos_texture,
            gpu::TextureViewDesc {
                name: "pos view",
                format: gpu::TextureFormat::Rgba32Float,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );
        let pos_sampler = ctx.create_sampler(gpu::SamplerDesc {
            name: "pos sampler",
            address_modes: Default::default(),
            mag_filter: gpu::FilterMode::Nearest,
            min_filter: gpu::FilterMode::Nearest,
            mipmap_filter: gpu::FilterMode::Nearest,
            ..Default::default()
        });

        let normal_texture = ctx.create_texture(gpu::TextureDesc {
            name: "normal texture",
            format: gpu::TextureFormat::Rgba32Float,
            size: extent,
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
        });
        let normal_view = ctx.create_texture_view(
            normal_texture,
            gpu::TextureViewDesc {
                name: "normal view",
                format: gpu::TextureFormat::Rgba32Float,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );
        let normal_sampler = ctx.create_sampler(gpu::SamplerDesc {
            name: "normal sampler",
            address_modes: Default::default(),
            mag_filter: gpu::FilterMode::Nearest,
            min_filter: gpu::FilterMode::Nearest,
            mipmap_filter: gpu::FilterMode::Nearest,
            ..Default::default()
        });

        let depth_textures = create_depth_textures(ctx, extent);
        let gbuffer = GBuffer {
            pos_texture,
            normal_texture,
            pos_view,
            normal_view,
            pos_sampler,
            normal_sampler,
            depth_textures,
        };
        gbuffer
    }
}

pub struct State {
    // pub light_pipeline: gpu::RenderPipeline,
    pub pipelines: Pipelines,
    pub command_encoder: gpu::CommandEncoder,
    pub ctx: gpu::Context,
    pub surface: gpu::Surface,
    pub prev_sync_point: Option<gpu::SyncPoint>,
    pub meshes: Vec<Mesh>,
    pub camera: Camera,
    pub retained_input: RetainedInput,
    pub g_buffer: GBuffer,
    pub screen_quad_buf: gpu::BufferPiece,
    pub depth_textures: DepthTextures,
}

#[derive(Default)]
pub struct RetainedInput {
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

        let depth_texture = ctx.create_texture(gpu::TextureDesc {
            name: "depth",
            format: gpu::TextureFormat::Depth32Float,
            size: screen_extent,
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
        });

        let depth_view = ctx.create_texture_view(
            depth_texture,
            gpu::TextureViewDesc {
                name: "depth view",
                format: gpu::TextureFormat::Depth32Float,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );

        let depth_sampler = ctx.create_sampler(gpu::SamplerDesc {
            name: "depth sampler",
            compare: Some(gpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let light_shader_source = std::fs::read_to_string("src/light_shader.wgsl").unwrap();
        let light_shader = ctx.create_shader(gpu::ShaderDesc {
            source: &light_shader_source,
        });

        let light_pipeline = ctx.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "light",
            // data_layouts: &[&<Params as gpu::ShaderData>::layout()],
            data_layouts: &[&<LightParams as gpu::ShaderData>::layout()],
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
            fragment: light_shader.at("fs_main"),
            color_targets: &[gpu::ColorTargetState {
                format: surface.info().format,
                blend: Some(gpu::BlendState::REPLACE),
                write_mask: gpu::ColorWrites::default(),
            }],
        });

        // let vertex_buf = ctx.create_buffer(gpu::BufferDesc {
        //     name: "vertex buffer",
        //     size: (vertices.len() * std::mem::size_of::<Vertex>()) as u64,
        //     memory: gpu::Memory::Shared,
        // });
        // unsafe {
        //     std::ptr::copy_nonoverlapping(
        //         vertices.as_ptr(),
        //         vertex_buf.data() as *mut Vertex,
        //         vertices.len(),
        //     );
        // }

        // let indices = (0..vertices.len())
        //     .into_iter()
        //     .map(|a| a as u32)
        //     .collect::<Vec<_>>();
        // let index_buf = ctx.create_buffer(gpu::BufferDesc {
        //     name: "index buffer",
        //     size: (indices.len() * std::mem::size_of::<u32>()) as u64,
        //     memory: gpu::Memory::Shared,
        // });

        // unsafe {
        //     std::ptr::copy_nonoverlapping(
        //         indices.as_ptr(),
        //         index_buf.data() as *mut u32,
        //         indices.len(),
        //     );
        // }
        let mut meshes = vec![];

        let command_encoder = ctx.create_command_encoder(gpu::CommandEncoderDesc {
            name: "main",
            buffer_count: 1,
        });

        let sponza_vertices = load_sponza();
        let a = sponza_vertices.len() / 3;
        dbg!(a);
        let gpu_sponza = upload_vertices(sponza_vertices, &ctx);
        meshes.clear();
        meshes.push(gpu_sponza);

        let g_buffer = GBuffer::new(&ctx, width, height);

        let screen_quad_vertices = [
            vec3(-1.0, -1.0, 0.0),
            vec3(1.0, -1.0, 0.0),
            vec3(-1.0, 1.0, 0.0),
            vec3(1.0, -1.0, 0.0),
            vec3(1.0, 1.0, 0.0),
            vec3(-1.0, 1.0, 0.0),
        ]
        .map(|a| Vertex {
            pos: a.to_array(),
            normal: Default::default(),
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

        let depth_textures = create_depth_textures(&ctx, screen_extent);

        let pipelines = Pipelines::create_pipelines(&ctx, &surface).unwrap();

        Self {
            command_encoder,
            ctx,
            surface,
            prev_sync_point: None,
            meshes,
            camera: Camera::default_from_aspect(aspect),
            retained_input: Default::default(),
            g_buffer,
            screen_quad_buf: screen_quad_buf.into(),
            depth_textures,
            pipelines,
        }
    }

    pub fn render_depth_downsamples(&mut self) {
        for i in 1..self.depth_textures.texture_stuffs.len() {
            let texture_from = &self.depth_textures.texture_stuffs[i - 1];
            let texture_to = &self.depth_textures.texture_stuffs[i];

            if let mut depth_downsample_pass = self.command_encoder.render(
                format!("depth downsample {i}").as_str(),
                gpu::RenderTargetSet {
                    colors: &[],
                    depth_stencil: Some(gpu::RenderTarget {
                        view: texture_to.view,
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                        finish_op: gpu::FinishOp::Store,
                    }),
                },
            ) {
                let mut rc = depth_downsample_pass.with(&self.pipelines.depth_downsample);
                rc.bind(
                    0,
                    &DepthDownsampleParams {
                        depth_from: texture_from.view,
                        depth_from_sampler: texture_from.sampler,
                    },
                );
                rc.bind_vertex(0, self.screen_quad_buf);
                let num_quad_vertices = 6;
                rc.draw(0, num_quad_vertices as _, 0, 1);
            }

            // self.command_encoder.present(frame);
        }
    }

    pub fn render(&mut self) {
        self.command_encoder.start();
        let depth_texture_stuff = &self.g_buffer.depth_textures.texture_stuffs[0];
        if let mut geometry_pass = self.command_encoder.render(
            "geometry",
            gpu::RenderTargetSet {
                colors: &[
                    gpu::RenderTarget {
                        view: self.g_buffer.pos_view,
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                        finish_op: gpu::FinishOp::Store,
                    },
                    gpu::RenderTarget {
                        view: self.g_buffer.normal_view,
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                        finish_op: gpu::FinishOp::Store,
                    },
                ],
                depth_stencil: Some(gpu::RenderTarget {
                    view: depth_texture_stuff.view,
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                    finish_op: gpu::FinishOp::Store,
                }),
            },
        ) {
            let mut rc = geometry_pass.with(&self.pipelines.geometry);

            rc.bind(
                0,
                &Params {
                    globals: Globals {
                        mvp_transform: self.camera.vp().to_cols_array_2d(),
                        cam_pos: self.camera.pos.to_array(),
                        cam_dir: self.camera.right_forward_up()[1].to_array(),
                        pad: [0; 2],
                    },
                    depth_view: depth_texture_stuff.view,
                    depth_sampler: depth_texture_stuff.sampler,
                },
            );

            for mesh in self.meshes.iter() {
                rc.bind_vertex(0, mesh.vertex_buf);
                rc.draw(0, mesh.num_vertices as _, 0, 1);
            }
        }

        self.render_depth_downsamples();
        let frame = self.surface.acquire_frame();
        self.command_encoder.init_texture(frame.texture());
        let depth_texture_stuff = &self.g_buffer.depth_textures.texture_stuffs[0];
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
            rc.bind(
                0,
                &LightParams {
                    pos_view: self.g_buffer.pos_view,
                    pos_sampler: self.g_buffer.pos_sampler,
                    normal_view: self.g_buffer.normal_view,
                    normal_sampler: self.g_buffer.normal_sampler,
                    depth_view: depth_texture_stuff.view,
                    depth_sampler: depth_texture_stuff.sampler,
                    // near: todo!(),
                    // far: todo!(),
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

        let speed = 0.01;
        let angle_speed = 0.003;

        for key in self.retained_input.held_keys.iter() {
            match key {
                winit::keyboard::KeyCode::KeyW => {
                    self.camera.pos += f * speed;
                }
                winit::keyboard::KeyCode::KeyA => {
                    self.camera.pos -= r * speed;
                }
                winit::keyboard::KeyCode::KeyS => {
                    self.camera.pos -= f * speed;
                }
                winit::keyboard::KeyCode::KeyD => {
                    self.camera.pos += r * speed;
                }
                winit::keyboard::KeyCode::KeyQ => {
                    self.camera.pos -= u * speed;
                }
                winit::keyboard::KeyCode::KeyE => {
                    self.camera.pos += u * speed;
                }

                // angle
                winit::keyboard::KeyCode::KeyI => {
                    self.camera.pitch += angle_speed;
                }
                winit::keyboard::KeyCode::KeyJ => {
                    self.camera.yaw += angle_speed;
                }
                winit::keyboard::KeyCode::KeyK => {
                    self.camera.pitch -= angle_speed;
                }
                winit::keyboard::KeyCode::KeyL => {
                    self.camera.yaw -= angle_speed;
                }
                _ => {}
            }
        }
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
    // pub fn to_vp(&self) -> glam::Mat4 {
    // glam::Mat4::perspective_rh(self.fov_rad,self.aspect , , )
    // }

    pub fn view(&self) -> glam::Mat4 {
        let rot_x = Quat::from_axis_angle(Vec3::X, self.pitch);
        let rot_y = Quat::from_axis_angle(Vec3::Y, self.yaw);
        let rot = rot_y * rot_x;

        let pos = Vec3::from_array(self.pos.to_array());
        let pos = Vec3::from_array(self.pos.to_array());
        let view = Mat4::from_scale_rotation_translation(Vec3A::ONE.into(), rot, pos).inverse();
        view
    }

    pub fn projection(&self) -> glam::Mat4 {
        glam::Mat4::perspective_rh(self.fov_rad, self.aspect, 0.001, 100.0)
    }

    pub fn default_from_aspect(aspect: f32) -> Self {
        Self {
            pos: Vec3A::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            fov_rad: TAU / 4.0,
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
}
pub fn load_sponza() -> Vec<Vertex> {
    dbg!("loading sponza");
    let path = std::path::Path::new("src/assets/sponza/sponza.obj");
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
                pos: pos.to_array(),
                normal: n.to_array(),
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
            pos: v.to_array(),
            normal: normals[i / 3].to_array(),
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
                        for (i, x) in rest.split(" ").enumerate() {
                            if let Ok(x) = x.parse() {
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
    let window_attributes = winit::window::Window::default_attributes().with_title("ssao");

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
                            state.retained_input.held_keys.insert(key_code);
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
                        state.recreate_pipelines_if_required();

                        // state.camera.pos -= 0.0001 * Vec3A::Z;
                        // state.camera.yaw += 0.0001;

                        let [r, f, u] = state.camera.right_forward_up();
                        // state.camera.yaw = TAU / 4.0;
                        // state.camera.pos += 0.001 * f;
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
