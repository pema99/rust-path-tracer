#![feature(int_roundings)]

//pub mod window;
pub mod trace;
pub mod bvh;
pub mod atlas;
pub mod asset;

//use window::open_window;

//fn main() {
    //    open_window();
    //}
    
use std::{sync::{Arc, atomic::{AtomicBool, AtomicU32, Ordering}}, time::Instant};

use eframe::{
    egui_wgpu::{self, wgpu},
    wgpu::util::DeviceExt,
};
use glam::{Mat3, Vec3};
use parking_lot::RwLock;
use shared_structs::TracingConfig;

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

fn start_render(app_state: &AppState, on_render: impl Fn() + Send + Sync + 'static) {
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
            on_render
        );
    });
}

fn on_gui(app_state: &mut AppState, egui_ctx: &egui::Context, frame: &eframe::Frame) {
    egui::Window::new("Settings").show(egui_ctx, |ui| {
        if app_state.running.load(Ordering::Relaxed) {
            if ui.button("Stop").clicked() {
                //display.gl_window().window().set_resizable(true);
                app_state.running.store(false, Ordering::Relaxed);
            }
        } else {
            if ui.button("Start").clicked() {
                //display.gl_window().window().set_resizable(false);
                let size = frame.info().window_info.size;
                app_state.initialize_for_new_render(size.x as u32, size.y as u32);
                let egui_ctx = egui_ctx.clone();
                start_render(&app_state, move || egui_ctx.request_repaint());
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

fn handle_input(app_state: &mut AppState, ui: &egui::Ui) {
    if app_state.last_input.elapsed().as_millis() < 16 {
        return;
    }
    app_state.last_input = Instant::now();

    if ui.input().pointer.primary_down() {
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
    if ui.input().key_down(egui::Key::A) {
        config.cam_position -= right.extend(0.0) * 0.1;
    }
    if ui.input().key_down(egui::Key::E) {
        config.cam_position.y += 0.1;
    }
    if ui.input().key_down(egui::Key::Q) {
        config.cam_position.y -= 0.1;
    }

    let delta = ui.input().pointer.delta();
    config.cam_rotation.x += delta.y * 0.005;
    config.cam_rotation.y += delta.x * 0.005;
}

pub struct Custom3d {
    app_state: AppState,
}

impl Custom3d {
    pub fn new<'a>(cc: &'a eframe::CreationContext<'a>) -> Self {
        // Get the WGPU render state from the eframe creation context. This can also be retrieved
        // from `eframe::Frame` when you don't have a `CreationContext` available.
        let render_state = cc.wgpu_render_state.as_ref().expect("WGPU enabled");

        let device = &render_state.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("../custom3d_wgpu_shader.wgsl").into()),
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
                targets: &[Some(render_state.target_format.into())],
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

        let screen_width = cc.integration_info.window_info.size.x as u32;
        let screen_height = cc.integration_info.window_info.size.y as u32;
        let render_buffer = device.create_buffer(&wgpu::BufferDescriptor { 
            label: None, 
            size: (screen_width * screen_height * 3 * 4) as u64,
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

        render_state
            .renderer
            .write()
            .paint_callback_resources
            .insert(TriangleRenderResources {
                pipeline,
                bind_group,
                uniform_buffer,
                render_buffer,
            });

        Self { app_state: AppState::new(screen_width, screen_height) }
    }
}

impl eframe::App for Custom3d {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        if self.app_state.interacting.load(Ordering::Relaxed) {
            ctx.request_repaint();
        }

        egui::CentralPanel::default().frame(egui::Frame::default().inner_margin(egui::Vec2::ZERO)).show(ctx, |ui| {
            on_gui(&mut self.app_state, ctx, frame);
            handle_input(&mut self.app_state, ui);
            self.custom_painting(ui);
        });
    }
}

impl Custom3d {
    fn custom_painting(&mut self, ui: &mut egui::Ui) {
        let (rect, _) =
            ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());

        // The callback function for WGPU is in two stages: prepare, and paint.
        let framebuffer = self.app_state.framebuffer.clone();
        let width = self.app_state.config.read().width;
        let height = self.app_state.config.read().height;
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
    }
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

fn main() {
    let mut native_options = eframe::NativeOptions::default();
    native_options.renderer = eframe::Renderer::Wgpu;
    eframe::run_native("rust-path-tracer", native_options, Box::new(|cc| Box::new(Custom3d::new(cc))));
}