pub use blade_graphics as gpu;
use bytemuck::{Pod, Zeroable};
pub use glam::*;

pub const PI: f32 = 3.14159265358979323846264338327950288;
pub const TAU: f32 = 2.0 * PI;

#[repr(C)]
#[derive(blade_macros::ShaderData, Clone, Copy, Pod, Zeroable)]
pub struct Globals {
    mvp_transform: [[f32; 4]; 4],
}

#[derive(blade_macros::Vertex)]
pub struct Vertex {
    pos: [f32; 3],
}

pub struct Mesh {
    vertex_buf: gpu::BufferPiece,
}

pub struct Camera {
    pos: Vec3A,
    x_angle: f32,
    y_angle: f32,
    fov_rad: f32,
    aspect: f32,
}

pub struct Cube {}

pub struct State {
    pipeline: gpu::RenderPipeline,
    command_encoder: gpu::CommandEncoder,
    ctx: gpu::Context,
    surface: gpu::Surface,
    prev_sync_point: Option<gpu::SyncPoint>,
    meshes: Vec<Mesh>,
    camera: Camera,
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
        let aspect = width as f32 / height as f32;
        let surface = ctx
            .create_surface_configured(
                window,
                gpu::SurfaceConfig {
                    size: gpu::Extent {
                        width,
                        height,
                        depth: 1,
                    },
                    usage: gpu::TextureUsage::TARGET,
                    display_sync: gpu::DisplaySync::Recent,
                    ..Default::default()
                },
            )
            .unwrap();

        let shader_source = std::fs::read_to_string("src/shader.wgsl").unwrap();
        let shader = ctx.create_shader(gpu::ShaderDesc {
            source: &shader_source,
        });

        let pipeline = ctx.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "main",
            data_layouts: &[&<Globals as gpu::ShaderData>::layout()],
            vertex: shader.at("vs_main"),
            vertex_fetches: &[gpu::VertexFetchState {
                layout: &<Vertex as gpu::Vertex>::layout(),
                instanced: false,
            }],
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            fragment: shader.at("fs_main"),
            color_targets: &[gpu::ColorTargetState {
                format: surface.info().format,
                blend: Some(gpu::BlendState::ALPHA_BLENDING),
                write_mask: gpu::ColorWrites::default(),
            }],
        });

        let extent = gpu::Extent {
            width: 1,
            height: 1,
            depth: 1,
        };
        let texture = ctx.create_texture(gpu::TextureDesc {
            name: "texture",
            format: gpu::TextureFormat::Rgba8Unorm,
            size: extent,
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::RESOURCE | gpu::TextureUsage::COPY,
        });

        let view = ctx.create_texture_view(
            texture,
            gpu::TextureViewDesc {
                name: "view",
                format: gpu::TextureFormat::Rgba8Unorm,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );

        let upload_buffer = ctx.create_buffer(gpu::BufferDesc {
            name: "staging",
            size: (extent.width * extent.height) as u64 * 4,
            memory: gpu::Memory::Upload,
        });

        let texture_data = [0xFFu8; 4];
        unsafe {
            std::ptr::copy_nonoverlapping(
                texture_data.as_ptr(),
                upload_buffer.data(),
                texture_data.len(),
            );
        }

        ctx.sync_buffer(upload_buffer);

        let sampler = ctx.create_sampler(gpu::SamplerDesc {
            name: "main",
            ..Default::default()
        });

        let vertices = [
            Vertex {
                pos: [0.2, 0.2, 0.0],
            },
            Vertex {
                pos: [0.8, 0.2, 0.0],
            },
            Vertex {
                pos: [0.5, 0.8, 0.0],
            },
        ];

        let vertex_buf = ctx.create_buffer(gpu::BufferDesc {
            name: "vertex buffer",
            size: (vertices.len() * std::mem::size_of::<Vertex>()) as u64,
            memory: gpu::Memory::Shared,
        });

        let meshes = vec![Mesh {
            vertex_buf: vertex_buf.into(),
        }];

        unsafe {
            std::ptr::copy_nonoverlapping(
                vertices.as_ptr(),
                vertex_buf.data() as *mut Vertex,
                vertices.len(),
            );
        }

        ctx.sync_buffer(vertex_buf);

        dbg!("hi");

        let mut command_encoder = ctx.create_command_encoder(gpu::CommandEncoderDesc {
            name: "main",
            buffer_count: 1,
        });

        ctx.destroy_buffer(upload_buffer);

        Self {
            pipeline,
            command_encoder,
            ctx,
            surface,
            prev_sync_point: None,
            meshes,
            camera: Camera::default_from_aspect(aspect),
        }
    }

    pub fn render(&mut self) {
        let frame = self.surface.acquire_frame();
        self.command_encoder.start();
        self.command_encoder.init_texture(frame.texture());
        if let mut pass = self.command_encoder.render(
            "main",
            gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: frame.texture_view(),
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: None,
            },
        ) {
            let mut rc = pass.with(&self.pipeline);

            // let vp = glam::Mat4::perspective_rh(, , , )

            let p = Vec4::ONE;
            let vp = self.camera.vp();

            rc.bind(
                0,
                &Globals {
                    mvp_transform: vp.to_cols_array_2d(),
                },
            );

            let q = vp * p;
            dbg!(q);

            for mesh in self.meshes.iter() {
                // rc.bind(1, )
                // rc.bind(0, )
                rc.bind_vertex(0, mesh.vertex_buf);
                rc.draw(0, 3, 0, 1);
                dbg!("draw here");
            }
        }
        self.command_encoder.present(frame);
        let sync_point = self.ctx.submit(&mut self.command_encoder);
        if let Some(sp) = self.prev_sync_point.take() {
            self.ctx.wait_for(&sp, !0);
        }
        self.prev_sync_point = Some(sync_point);
    }
}

impl Camera {
    // pub fn to_vp(&self) -> glam::Mat4 {
    // glam::Mat4::perspective_rh(self.fov_rad,self.aspect , , )
    // }

    pub fn view(&self) -> glam::Mat4 {
        let rot_x = Quat::from_axis_angle(Vec3::X, self.y_angle.to_radians());
        let rot_z = Quat::from_axis_angle(Vec3::Z, self.x_angle);
        let rot = rot_z * rot_x;
        let proj = Mat4::from_scale_rotation_translation(Vec3A::ONE.into(), rot, self.pos.into());
        proj.inverse()
    }

    pub fn projection(&self) -> glam::Mat4 {
        glam::Mat4::perspective_rh(self.fov_rad, self.aspect, 0.1, 100.0)
    }

    pub fn default_from_aspect(aspect: f32) -> Self {
        Self {
            pos: Vec3A::ZERO,
            x_angle: 0.0,
            y_angle: 0.0,
            fov_rad: TAU / 4.0,
            aspect,
        }
    }

    pub fn vp(&self) -> glam::Mat4 {
        self.projection() * self.view()
    }
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
                                state: winit::event::ElementState::Pressed,
                                ..
                            },
                        ..
                    } => match key_code {
                        winit::keyboard::KeyCode::Space => {
                            dbg!("Hello");
                        }
                        _ => {}
                    },
                    winit::event::WindowEvent::CloseRequested => {
                        target.exit();
                    }
                    winit::event::WindowEvent::RedrawRequested => {
                        state.render();
                    }
                    _ => {}
                },
                _ => {}
            }
        })
        .unwrap();
}
