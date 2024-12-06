use nanorand::Rng;
use std::io::BufRead;

pub use blade_graphics as gpu;
use bytemuck::{Pod, Zeroable};
pub use glam::*;

pub const PI: f32 = 3.14159265358979323846264338327950288;
pub const TAU: f32 = 2.0 * PI;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Globals {
    mvp_transform: [[f32; 4]; 4],
}

#[derive(blade_macros::ShaderData)]
pub struct Params {
    pub globals: Globals,
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

pub struct Cube {}

pub struct State {
    pub pipeline: gpu::RenderPipeline,
    pub command_encoder: gpu::CommandEncoder,
    pub ctx: gpu::Context,
    pub surface: gpu::Surface,
    pub prev_sync_point: Option<gpu::SyncPoint>,
    pub meshes: Vec<Mesh>,
    pub camera: Camera,
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
            data_layouts: &[&<Params as gpu::ShaderData>::layout()],
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

        // let upload_buffer = ctx.create_buffer(gpu::BufferDesc {
        //     name: "staging",
        //     size: (extent.width * extent.height) as u64 * 4,
        //     memory: gpu::Memory::Upload,
        // });

        // let texture_data = [0xFFu8; 4];
        // unsafe {
        //     std::ptr::copy_nonoverlapping(
        //         texture_data.as_ptr(),
        //         upload_buffer.data(),
        //         texture_data.len(),
        //     );
        // }

        // ctx.sync_buffer(upload_buffer);

        let sampler = ctx.create_sampler(gpu::SamplerDesc {
            name: "main",
            ..Default::default()
        });

        let mut vertices = [
            vec3a(0.2, 0.2, 1.0),
            vec3a(0.8, 0.2, 1.0),
            vec3a(0.5, 0.8, 1.0),
            vec3a(-0.2, -0.2, 1.0),
            vec3a(-0.8, -0.2, 1.0),
            vec3a(-0.5, -0.8, 1.0),
        ];

        // let mut vertices = vertices.to_vec();
        // vertices.extend(v);

        let normals = vertices
            .chunks(3)
            .map(|c| {
                let v0 = c[0];
                let v1 = c[1];
                let v2 = c[2];
                let n = (v1 - v0).cross(v2 - v0).normalize();
                n
            })
            .collect::<Vec<_>>();

        let vertices = vertices
            .into_iter()
            .enumerate()
            .map(|(i, pos)| Vertex {
                pos: pos.to_array(),
                normal: normals[i / 3].to_array(),
            })
            .collect::<Vec<_>>();

        let mut vertices = vec![];
        let mut rng = nanorand::wyrand::WyRand::new();
        for i in 0..10 * 3 {
            let r = rng.rand();
            let r = r.map(|a| a as f32 / u8::MAX as f32);
            let mut v = vec3a(r[0], r[1], r[2]);
            v = v;
            v = v - 0.5;
            v = v * 5.0;

            v = v / v.max_element();
            let vertex = Vertex {
                pos: v.into(),
                normal: [r[3], 0.0, 0.0],
            };

            vertices.push(vertex);
        }

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
        let index_buf = ctx.create_buffer(gpu::BufferDesc {
            name: "index buffer",
            size: (vertices.len() * std::mem::size_of::<u16>()) as u64,
            memory: gpu::Memory::Shared,
        });

        let indices = (0..vertices.len())
            .into_iter()
            .map(|a| a as u16)
            .collect::<Vec<_>>();
        unsafe {
            std::ptr::copy_nonoverlapping(
                indices.as_ptr(),
                index_buf.data() as *mut u16,
                indices.len(),
            );
        }
        let mut meshes = vec![];

        let test_mesh = Mesh {
            vertex_buf: vertex_buf.into(),
            num_vertices: vertices.len(),
            index_buf: Some(index_buf.into()),
            num_indices: indices.len(),
        };
        meshes.push(test_mesh);

        ctx.sync_buffer(vertex_buf);
        ctx.sync_buffer(index_buf);

        let mut command_encoder = ctx.create_command_encoder(gpu::CommandEncoderDesc {
            name: "main",
            buffer_count: 2,
        });

        // ctx.destroy_buffer(upload_buffer);

        // let sponza_mesh = load_sponza();
        // let gpu_sponza = upload_mesh(&ctx, sponza_mesh);
        // meshes.push(gpu_sponza);

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

            let vp = self.camera.vp();

            rc.bind(
                0,
                &Params {
                    globals: Globals {
                        mvp_transform: vp.to_cols_array_2d(),
                    },
                },
            );

            // let q = vp * p;
            // let q = q.xyz() / q.w;

            // dbg!(q);

            for mesh in self.meshes.iter() {
                rc.bind_vertex(0, mesh.vertex_buf);
                if false {
                    if let Some(index_buf) = mesh.index_buf {
                        rc.draw_indexed(
                            index_buf,
                            gpu::IndexType::U16,
                            mesh.num_indices as _,
                            0,
                            0,
                            1,
                        );
                    }
                } else {
                    rc.draw(0, mesh.num_vertices as _, 0, 1);
                }
                // rc.bind(1, )
                // rc.bind(0, )
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
        let rot_x = Quat::from_axis_angle(Vec3::X, self.pitch.to_radians());
        let rot_y = Quat::from_axis_angle(Vec3::Y, self.yaw);
        let rot = rot_y * rot_x;

        let pos = Vec3::from_array(self.pos.to_array());
        let pos = Vec3::from_array(self.pos.to_array());
        let view = Mat4::from_scale_rotation_translation(Vec3A::ONE.into(), rot, -pos);
        view
    }

    pub fn projection(&self) -> glam::Mat4 {
        glam::Mat4::perspective_rh(self.fov_rad, self.aspect, 0.1, 100.0)
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
}
pub fn load_sponza() -> CpuMesh {
    dbg!("loading sponza");
    let path = std::path::Path::new("src/assets/sponza/sponza.obj");
    parse_obj_file(path)
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
    let indices = indices.iter().map(|idx| *idx as u64).collect::<Vec<_>>();
    let index_buf = ctx.create_buffer(gpu::BufferDesc {
        name: "index buffer",
        size: (indices.len() * std::mem::size_of::<u64>()) as u64,
        memory: gpu::Memory::Shared,
    });

    unsafe {
        std::ptr::copy_nonoverlapping(
            indices.as_ptr(),
            index_buf.data() as *mut u64,
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
                        dbg!("closing");
                        target.exit();
                    }
                    winit::event::WindowEvent::RedrawRequested => {
                        // state.camera.pos -= 0.0001 * Vec3A::Z;
                        state.camera.yaw += 0.0001;
                        state.render();
                    }
                    _ => {}
                },
                _ => {}
            }
        })
        .unwrap();
}
