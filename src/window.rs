
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::{iter, sync::Arc};
use std::time::Instant;

use ::egui::FontDefinitions;
use egui_wgpu::renderer::ScreenDescriptor;
use egui_winit_platform::{Platform, PlatformDescriptor};
use wgpu::util::DeviceExt;
use winit::event::Event::*;
use winit::event_loop::ControlFlow;
const INITIAL_WIDTH: u32 = 1920;
const INITIAL_HEIGHT: u32 = 1080;

use glam::{Mat3, Vec3};
use parking_lot::RwLock;
use shared_structs::TracingConfig;
use winit::window::Window;

use crate::trace::trace;

struct AppState {
    framebuffer: Arc<RwLock<Vec<f32>>>,
    running: Arc<AtomicBool>,
    samples: Arc<AtomicU32>,
    denoise: Arc<AtomicBool>,
    sync_rate: Arc<AtomicU32>,
    interacting: Arc<AtomicBool>,
    config: Arc<RwLock<TracingConfig>>,
    last_input: Instant,
}

fn make_view_dependent_state(
    width: u32,
    height: u32,
    config: Option<TracingConfig>,
) -> (
    Arc<RwLock<TracingConfig>>,
    Arc<RwLock<Vec<f32>>>,
) {
    let config = Arc::new(RwLock::new(TracingConfig {
        width,
        height,
        ..config.unwrap_or_default()
    }));
    let data_size = width as usize * height as usize * 3;
    let framebuffer = Arc::new(RwLock::new(vec![0.0; data_size]));
    (config, framebuffer)
}

impl AppState {
    fn new(width: u32, height: u32) -> Self {
        let (config, framebuffer) = make_view_dependent_state(width, height, None);
        let running = Arc::new(AtomicBool::new(false));
        let samples = Arc::new(AtomicU32::new(0));
        let denoise = Arc::new(AtomicBool::new(false));
        let sync_rate = Arc::new(AtomicU32::new(128));
        let interacting = Arc::new(AtomicBool::new(false));
        Self {
            framebuffer,
            running,
            samples,
            denoise,
            sync_rate,
            interacting,
            config,
            last_input: Instant::now(),
        }
    }

    fn initialize_for_new_render(&mut self, width: u32, height: u32) {
        let (config, framebuffer) =
            make_view_dependent_state(width, height, Some(self.config.read().clone()));
        self.config = config;
        self.framebuffer = framebuffer;
        self.samples.store(0, Ordering::Relaxed);
    }
}

fn start_render(app_state: &AppState) {
    app_state.running.store(true, Ordering::Relaxed);
    let data = app_state.framebuffer.clone();
    let running = app_state.running.clone();
    let samples = app_state.samples.clone();
    let denoise = app_state.denoise.clone();
    let sync_rate = app_state.sync_rate.clone();
    let interacting = app_state.interacting.clone();
    let config = app_state.config.clone();
    std::thread::spawn(move || {
        trace(
            data,
            running,
            samples,
            denoise,
            sync_rate,
            interacting,
            config,
        );
    });
}

fn on_gui(app_state: &mut AppState, egui_ctx: &egui::Context, window: &Window) {
    egui::Window::new("Settings").show(egui_ctx, |ui| {
        if app_state.running.load(Ordering::Relaxed) {
            if ui.button("Stop").clicked() {
                window.set_resizable(true);
                app_state.running.store(false, Ordering::Relaxed);
            }
        } else {
            if ui.button("Start").clicked() {
                window.set_resizable(false);
                let size = window.inner_size();
                app_state.initialize_for_new_render(size.width, size.height);
                start_render(&app_state);
            }
        }

        #[cfg(feature = "oidn")]
        {
            let mut denoise_checked = app_state.denoise.load(Ordering::Relaxed);
            if ui.checkbox(&mut denoise_checked, "Denoise").changed() {
                app_state.denoise.store(denoise_checked, Ordering::Relaxed);
            }
        }

        {
            let mut sync_rate = app_state.sync_rate.load(Ordering::Relaxed);
            if ui
                .add(egui::Slider::new(&mut sync_rate, 1..=1024).text("Sync rate"))
                .changed()
            {
                app_state.sync_rate.store(sync_rate, Ordering::Relaxed);
            }
        }

        ui.label(format!(
            "Samples: {}",
            app_state.samples.load(Ordering::Relaxed)
        ));
    });
}

fn handle_input(app_state: &mut AppState, mouse_delta: &mut (f32, f32), ui: &egui::Ui) {
    if app_state.last_input.elapsed().as_millis() < 16 {
        return;
    }
    app_state.last_input = Instant::now();

    if ui.input().pointer.secondary_down() {
        app_state.interacting.store(true, Ordering::Relaxed);
    } else {
        app_state.interacting.store(false, Ordering::Relaxed);
    }

    let mut config = app_state.config.write();

    let mut forward = Vec3::new(0.0, 0.0, 1.0);
    let mut right = Vec3::new(1.0, 0.0, 0.0);
    let euler_mat = Mat3::from_rotation_y(config.cam_rotation.y) * Mat3::from_rotation_x(config.cam_rotation.x);
    forward = euler_mat * forward;
    right = euler_mat * right;
    
    if ui.input().key_down(egui::Key::W) {
        config.cam_position += forward.extend(0.0) * 0.1;
    }
    if ui.input().key_down(egui::Key::S) {
        config.cam_position -= forward.extend(0.0) * 0.1;
    }
    if ui.input().key_down(egui::Key::D) {
        config.cam_position += right.extend(0.0) * 0.1;
    }
    if ui.input().key_down(egui::Key::A){
        config.cam_position -= right.extend(0.0) * 0.1;
    }
    if ui.input().key_down(egui::Key::E){
        config.cam_position.y += 0.1;
    }
    if ui.input().key_down(egui::Key::Q){
        config.cam_position.y -= 0.1;
    }

    config.cam_rotation.x += mouse_delta.1 * 0.005;
    config.cam_rotation.y += mouse_delta.0 * 0.005;
    *mouse_delta = (0.0, 0.0);
}

pub fn open_window() {
    let event_loop = winit::event_loop::EventLoopBuilder::<()>::with_user_event().build();
    let window = winit::window::WindowBuilder::new()
        .with_decorations(true)
        .with_resizable(true)
        .with_transparent(false)
        .with_title("rust-path-tracer")
        .with_inner_size(winit::dpi::PhysicalSize {
            width: INITIAL_WIDTH,
            height: INITIAL_HEIGHT,
        })
        .build(&event_loop)
        .unwrap();
    let mut app_state = AppState::new(INITIAL_WIDTH, INITIAL_HEIGHT);

    let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
    let surface = unsafe { instance.create_surface(&window) };

    // WGPU 0.11+ support force fallback (if HW implementation not supported), set it to true or false (optional).
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .unwrap();

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            features: wgpu::Features::default(),
            limits: wgpu::Limits::default(),
            label: None,
        },
        None,
    ))
    .unwrap();

    let size = window.inner_size();
    let surface_format = surface.get_supported_formats(&adapter)[0];
    let mut surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: size.width as u32,
        height: size.height as u32,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
    };
    surface.configure(&device, &surface_config);

    // We use the egui_winit_platform crate as the platform.
    let mut platform = Platform::new(PlatformDescriptor {
        physical_width: size.width as u32,
        physical_height: size.height as u32,
        scale_factor: window.scale_factor(),
        font_definitions: FontDefinitions::default(),
        style: Default::default(),
    });

    // We use the egui_wgpu_backend crate as the render backend.
    let mut egui_rpass = egui_wgpu::renderer::Renderer::new(&device, surface_format, None, 1);
    egui_rpass.paint_callback_resources.insert(create_render_resources(&device, surface_format, INITIAL_WIDTH, INITIAL_HEIGHT));

    let mut mouse_delta = (0.0, 0.0);
    let start_time = Instant::now();
    event_loop.run(move |event, _, control_flow| {
        // Pass the winit events to the platform integration.
        platform.handle_event(&event);

        match event {
            RedrawRequested(..) => {
                platform.update_time(start_time.elapsed().as_secs_f64());

                let output_frame = match surface.get_current_texture() {
                    Ok(frame) => frame,
                    Err(_) => { return; }
                };
                let output_view = output_frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                // Begin to draw the UI frame.
                platform.begin_frame();

                // Render here
                egui::CentralPanel::default().frame(egui::Frame::default().inner_margin(egui::Vec2::ZERO)).show(&platform.context(), |ui| {
                    on_gui(&mut app_state, &platform.context(), &window);
                    handle_input(&mut app_state, &mut mouse_delta, ui);
                    let (rect, _) =
                        ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());
                    let framebuffer = app_state.framebuffer.clone();
                    let width = app_state.config.read().width;
                    let height = app_state.config.read().height;
                    let cb = egui_wgpu::CallbackFn::new()
                        .prepare(move |device, queue, _encoder, typemap| {
                            let resources: &TriangleRenderResources = typemap.get().unwrap();
                            
                            resources.prepare(device, queue, &framebuffer, width, height);
                            vec![]
                        })
                        .paint(move |_info, rpass, typemap| {
                            let resources: &TriangleRenderResources = typemap.get().unwrap();
            
                            resources.paint(rpass);
                        });
            
                    let callback = egui::PaintCallback {
                        rect,
                        callback: Arc::new(cb),
                    };
            
                    ui.painter().add(callback);
                });
                

                // End the UI frame. We could now handle the output and draw the UI with the backend.
                let full_output = platform.end_frame(Some(&window));
                let paint_jobs = platform.context().tessellate(full_output.shapes);

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

                // Upload all resources for the GPU.
                let screen_descriptor = ScreenDescriptor {
                    size_in_pixels: [surface_config.width, surface_config.height],
                    pixels_per_point: window.scale_factor() as f32,
                };
                let tdelta: egui::TexturesDelta = full_output.textures_delta;
                for (id, image_delta) in &tdelta.set {
                    egui_rpass.update_texture(&device, &queue, *id, &image_delta);
                }
                egui_rpass.update_buffers(
                    &device,
                    &queue,
                    &mut encoder,
                    &paint_jobs,
                    &screen_descriptor,
                );

                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render Pass"),
                        color_attachments: &[
                            // This is what @location(0) in the fragment shader targets
                            Some(wgpu::RenderPassColorAttachment {
                                view: &output_view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color {
                                        r: 0.1,
                                        g: 0.2,
                                        b: 0.3,
                                        a: 1.0,
                                    }),
                                    store: true,
                                },
                            }),
                        ],
                        depth_stencil_attachment: None,
                    });
                    // Record all render passes.
                    egui_rpass.render(&mut render_pass, &paint_jobs, &screen_descriptor);
                }
                // Submit the commands.
                queue.submit(iter::once(encoder.finish()));

                // Redraw egui
                output_frame.present();

                for id in &tdelta.free {
                    egui_rpass.free_texture(&id);
                }
            }
            DeviceEvent { event: winit::event::DeviceEvent::MouseMotion { delta }, ..} => {
                if app_state.interacting.load(Ordering::Relaxed) {
                    mouse_delta.0 += delta.0 as f32;
                    mouse_delta.1 += delta.1 as f32;
                
                    // Set mouse position to center of screen
                    let size = window.inner_size();
                    let center = winit::dpi::LogicalPosition::new(size.width as f32 / 2.0, size.height as f32 / 2.0);
                    window.set_cursor_position(center).unwrap();
                }
            }
            MainEventsCleared | UserEvent(()) => {
                window.request_redraw();
            }
            WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::Resized(size) => {
                    if size.width > 0 && size.height > 0 {
                        surface_config.width = size.width;
                        surface_config.height = size.height;
                        surface.configure(&device, &surface_config);
                    }
                }
                winit::event::WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                _ => {}
            },
            _ => (),
        }
    });
}

struct TriangleRenderResources {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    render_buffer: wgpu::Buffer,
}

impl TriangleRenderResources {
    fn prepare(&self, _device: &wgpu::Device, queue: &wgpu::Queue, framebuffer: &Arc<RwLock<Vec<f32>>>, width: u32, height: u32) {
        let vec = framebuffer.read();
        queue.write_buffer(&self.render_buffer, 0, bytemuck::cast_slice(&vec));
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[width, height]));
    }

    fn paint<'rpass>(&'rpass self, rpass: &mut wgpu::RenderPass<'rpass>) {
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.draw(0..6, 0..1);
    }
}

fn create_render_resources(device: &wgpu::Device,format: wgpu::TextureFormat, width: u32, height: u32) -> TriangleRenderResources {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/render.wgsl").into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&[0.0]),
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::UNIFORM,
    });

    let render_buffer = device.create_buffer(&wgpu::BufferDescriptor { 
        label: None, 
        size: (width * height * 3 * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST |
               wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false 
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: render_buffer.as_entire_binding(),
        }],
    });

    TriangleRenderResources { pipeline, bind_group, uniform_buffer, render_buffer }
}