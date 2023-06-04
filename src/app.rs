
use std::num::NonZeroU32;
use std::sync::atomic::Ordering;
use std::time::Instant;
use std::{iter, sync::Arc};
use std::fmt::Debug;

use egui_wgpu::renderer::ScreenDescriptor;
use egui_winit_platform::Platform;
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

use glam::{Mat3, Vec3};
use shared_structs::NextEventEstimation;

use crate::trace::{trace_cpu, trace_gpu, TracingState};

#[repr(u32)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum Tonemapping {
    None,
    Reinhard,
    ACESNarkowicz,
    ACESNarkowiczOverexposed,
    ACESHill,
    Neutral,
    Uncharted,
}

impl Debug for Tonemapping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tonemapping::None => write!(f, "None"),
            Tonemapping::Reinhard => write!(f, "Reinhard"),
            Tonemapping::ACESNarkowicz => write!(f, "ACES (N)"),
            Tonemapping::ACESNarkowiczOverexposed => write!(f, "ACES (N, O)"),
            Tonemapping::ACESHill => write!(f, "ACES (H)"),
            Tonemapping::Neutral => write!(f, "Neutral"),
            Tonemapping::Uncharted => write!(f, "Uncharted"),
        }
    }
}

fn is_image(img: &str) -> bool {
    img.ends_with(".png")
    || img.ends_with(".jpg")
    || img.ends_with(".jpeg")
    || img.ends_with(".bmp")
    || img.ends_with(".tga")
    || img.ends_with(".hdr")
    || img.ends_with(".exr")
}

pub struct App {
    tracing_state: Arc<TracingState>,
    compute_join_handle: Option<std::thread::JoinHandle<()>>,

    use_cpu: bool,
    tonemapping: Tonemapping,
    selected_scene: String,
    selected_skybox: Option<String>,
    show_environment_window: bool,
    last_input: Instant,
    mouse_delta: (f32, f32),

    device: wgpu::Device,
    queue: wgpu::Queue,
    window: winit::window::Window,
    surface_format: wgpu::TextureFormat,
    surface_config: wgpu::SurfaceConfiguration,
    surface: wgpu::Surface,
    egui_renderer: egui_wgpu::renderer::Renderer,
}

impl App {
    pub fn new(window: winit::window::Window) -> Self {    
        let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
        let surface = unsafe { instance.create_surface(&window) };
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("Failed to creator wgpu adapter.");
    
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::default(),
                limits: wgpu::Limits::default(),
                label: None,
            },
            None,
        ))
        .expect("Failed to creator wgpu device.");

        let size = window.inner_size();
        let surface_format = surface.get_supported_formats(&adapter)[0];
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
        };
        surface.configure(&device, &surface_config);

        let egui_renderer = egui_wgpu::renderer::Renderer::new(&device, surface_format, None, 1);
        let tracing_state = Arc::new(TracingState::new(size.width, size.height));
        Self {
            tracing_state,
            last_input: Instant::now(),
            mouse_delta: (0.0, 0.0),
            device,
            queue,
            window,
            surface_config,
            surface,
            surface_format,
            egui_renderer,
            compute_join_handle: None,
            selected_scene: "scene.glb".to_string(),
            selected_skybox: None,
            tonemapping: Tonemapping::None,
            use_cpu: false,
            show_environment_window: false,
        }
    }

    pub fn window(&self) -> &winit::window::Window {
        &self.window
    }

    fn start_render(&mut self, continue_previous: bool) {
        if self.compute_join_handle.is_some() {
            self.stop_render();
        }

        self.window.set_resizable(false);
        let size = self.window.inner_size();
        
        if !continue_previous {
            let (config, framebuffer) = TracingState::make_view_dependent_state(size.width, size.height, Some(*self.tracing_state.config.read()));
            *self.tracing_state.config.write() = config;
            *self.tracing_state.framebuffer.write() = framebuffer;
            self.tracing_state.samples.store(0, Ordering::Relaxed);

            let render_resources = PaintCallbackResources::new(&self.device, self.surface_format, size.width, size.height);
            self.egui_renderer.paint_callback_resources.insert(render_resources);
        }
        self.tracing_state.running.store(true, Ordering::Relaxed);
        let tracing_state = self.tracing_state.clone();

        let use_cpu = self.use_cpu;
        let path = self.selected_scene.clone();
        let skybox_path = self.selected_skybox.clone();
        self.compute_join_handle = Some(std::thread::spawn(move || {
            let skybox_path_ref = skybox_path.as_ref().map(|s| s.as_str());
            if use_cpu {
                trace_cpu(&path, skybox_path_ref, tracing_state);
            } else {
                trace_gpu(&path, skybox_path_ref, tracing_state);
            }
        }));
    }

    fn stop_render(&mut self) {
        self.window.set_resizable(true);
        self.tracing_state.running.store(false, Ordering::Relaxed);

        if let Some(handle) = self.compute_join_handle.take() {
            handle.join().expect("Render thread died.");
            self.compute_join_handle = None;
        }
    }

    fn restart_current_render(&mut self, continue_previous: bool) {
        if self.compute_join_handle.is_some() {
            self.start_render(continue_previous);
        }
    }

    fn set_skybox(&mut self, skybox: &str) {
        self.selected_skybox = Some(skybox.to_string());
        self.tracing_state.config.write().has_skybox = 1;
        self.restart_current_render(false);
    }

    fn clear_skybox(&mut self) {
        self.selected_skybox = None;
        self.tracing_state.config.write().has_skybox = 0;
        self.restart_current_render(false);
    }

    fn set_scene(&mut self, scene: &str) {
        self.selected_scene = scene.to_string();
        self.start_render(false);
    }

    fn on_gui(&mut self, egui_ctx: &egui::Context) {
        self.on_settings_gui(egui_ctx);
        self.on_environment_gui(egui_ctx);
    }

    fn on_settings_gui(&mut self, egui_ctx: &egui::Context) {
        egui::Window::new("Settings").show(egui_ctx, |ui| {
            egui::Grid::new("MainGrid")
            .striped(true)
            .show(ui, |ui| {
                ui.vertical(|ui| {
                    ui.label(format!("Selected scene: {}", self.selected_scene));
                    ui.horizontal(|ui| {
                        if self.compute_join_handle.as_ref().map_or(false, |t| !t.is_finished()) {
                            if ui.button("Stop").clicked() {
                                self.stop_render();
                            }
                        } else {
                            if ui.button("Start").clicked() {
                                self.start_render(false);
                            }
                        }

                        if ui.button("Select scene").clicked() {
                            if let Some(path) = tinyfiledialogs::open_file_dialog("Select scene", "", None) {
                                if is_image(&path) {
                                    self.set_skybox(&path);
                                } else {
                                    self.set_scene(&path);
                                }
                            }
                        }

                        if ui.button("Save image").clicked() {
                            if let Some(resources) = self.egui_renderer.paint_callback_resources.get::<PaintCallbackResources>() {
                                let width = self.tracing_state.config.read().width;
                                let height = self.tracing_state.config.read().height;
                                resources.save_render(width, height, self.surface_format, &self.device, &self.queue);
                            }
                        }
                    });
                });
                ui.end_row();
                
                ui.horizontal(|ui| {
                    #[cfg(feature = "oidn")]
                    {
                        let mut denoise_checked = self.tracing_state.denoise.load(Ordering::Relaxed);
                        if ui.checkbox(&mut denoise_checked, "Denoise").changed() {
                            self.tracing_state.denoise.store(denoise_checked, Ordering::Relaxed);
                        }
                    }
    
                    let mut use_blue_noise = self.tracing_state.use_blue_noise.load(Ordering::Relaxed);
                    if ui.checkbox(&mut use_blue_noise, "Use blue noise").changed() {
                        self.tracing_state.use_blue_noise.store(use_blue_noise, Ordering::Relaxed);
                        self.tracing_state.dirty.store(true, Ordering::Relaxed);
                    }
                });
                ui.end_row();
    
                ui.horizontal(|ui| {
                    let mut config = self.tracing_state.config.write();
                    if ui.add(egui::DragValue::new(&mut config.min_bounces)).changed() {
                        if config.min_bounces > config.max_bounces {
                            config.max_bounces = config.min_bounces;
                        }
                        self.tracing_state.dirty.store(true, Ordering::Relaxed);
                    }
                    ui.label("Min bounces");
    
                    if ui.add(egui::DragValue::new(&mut config.max_bounces)).changed() {
                        if config.max_bounces < config.min_bounces {
                            config.min_bounces = config.max_bounces;
                        }
                        self.tracing_state.dirty.store(true, Ordering::Relaxed);
                    }
                    ui.label("Max bounces");
                });
                ui.end_row();

                let prev_nee_mode = NextEventEstimation::from_u32(self.tracing_state.config.read().nee);
                let mut nee_mode = prev_nee_mode;
                egui::ComboBox::from_label("Next event estimation")
                    .selected_text(format!("{:?}", nee_mode))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut nee_mode, NextEventEstimation::None, "None");
                        ui.selectable_value(&mut nee_mode, NextEventEstimation::MultipleImportanceSampling, "Multiple importance sampling");
                        ui.selectable_value(&mut nee_mode, NextEventEstimation::DirectLightSampling, "Direct light sampling only");
                    });
                if nee_mode != prev_nee_mode {
                    self.tracing_state.config.write().nee = nee_mode.to_u32();
                    self.tracing_state.dirty.store(true, Ordering::Relaxed);
                }
                ui.end_row();

                {
                    let mut config = self.tracing_state.config.write();
                    if ui.add(egui::Slider::new(&mut config.specular_weight_clamp.x, 0.0..=1.0).text("Min specular")).changed() {
                        if config.specular_weight_clamp.x > config.specular_weight_clamp.y {
                            config.specular_weight_clamp.y = config.specular_weight_clamp.x;
                        }
                        self.tracing_state.dirty.store(true, Ordering::Relaxed);
                    }
                    ui.end_row();

                    if ui.add(egui::Slider::new(&mut config.specular_weight_clamp.y, 0.0..=1.0).text("Max specular")).changed() {
                        if config.specular_weight_clamp.x > config.specular_weight_clamp.y {
                            config.specular_weight_clamp.x = config.specular_weight_clamp.y;
                        }
                        self.tracing_state.dirty.store(true, Ordering::Relaxed);
                    }
                    ui.end_row();
                }

                egui::ComboBox::from_label("Tonemapping operator")
                    .selected_text(format!("{:?}", self.tonemapping))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.tonemapping, Tonemapping::None, "None");
                        ui.selectable_value(&mut self.tonemapping, Tonemapping::ACESNarkowicz, "ACES (Narkowicz)");
                        ui.selectable_value(&mut self.tonemapping, Tonemapping::ACESNarkowiczOverexposed, "ACES (Narkowicz, overexposed)");
                        ui.selectable_value(&mut self.tonemapping, Tonemapping::ACESHill, "ACES (Hill)");
                        ui.selectable_value(&mut self.tonemapping, Tonemapping::Neutral, "Neutral");
                        ui.selectable_value(&mut self.tonemapping, Tonemapping::Reinhard, "Reinhard");
                        ui.selectable_value(&mut self.tonemapping, Tonemapping::Uncharted, "Uncharted");
                    });
                ui.end_row();

                if ui.button("Environment settings").clicked() {
                    self.show_environment_window = !self.show_environment_window;
                }
                ui.end_row();

                ui.separator();
                ui.end_row();

                egui::ComboBox::from_label("Compute device")
                    .selected_text(if self.use_cpu { "CPU" } else { "GPU" })
                    .show_ui(ui, |ui| {
                        if ui.selectable_label(!self.use_cpu, "GPU").clicked() { 
                            self.use_cpu = false;
                            self.restart_current_render(true);
                        };
                        if ui.selectable_label(self.use_cpu, "CPU").clicked() {
                            self.use_cpu = true;
                            self.restart_current_render(true);
                        };
                    });
                ui.end_row();

                let mut sync_rate = self.tracing_state.sync_rate.load(Ordering::Relaxed);
                if ui.add_enabled(!self.use_cpu, egui::Slider::new(&mut sync_rate, 1..=256).text("GPU sync rate")).changed() {
                    self.tracing_state.sync_rate.store(sync_rate, Ordering::Relaxed);
                }
                ui.end_row();
        
                ui.label(format!(
                    "Samples: {}",
                    self.tracing_state.samples.load(Ordering::Relaxed)
                ));
                ui.end_row();
            });
        });
    }

    fn on_environment_gui(&mut self, egui_ctx: &egui::Context) {
        let mut show_environment_window = self.show_environment_window;
        egui::Window::new("Environment").open(&mut show_environment_window).show(egui_ctx, |ui| {
            let mouse_down = ui.input().pointer.primary_down();
            let sun_direction = self.tracing_state.config.read().sun_direction;
            {
                let skybox_name = self.selected_skybox.as_ref().map(|s| s.as_ref()).unwrap_or("Procedural");
                ui.label(format!("Selected skybox: {}", skybox_name));
            }
            ui.horizontal(|ui| {
                if ui.button("Select skybox").clicked() {
                    if let Some(path) = tinyfiledialogs::open_file_dialog("Select skybox", "", None) {
                        self.set_skybox(&path);
                    }
                }
                if ui.button("Reset skybox").clicked() {
                    self.clear_skybox();
                }
            });

            let mut sun_intensity = sun_direction.w;
            if ui.add(egui::Slider::new(&mut sun_intensity, 0.0..=50.0).text("Sun intensity")).changed() {
                self.tracing_state.config.write().sun_direction.w = sun_intensity;
                self.tracing_state.dirty.store(true, Ordering::Relaxed);
            }
            ui.end_row();

            egui::plot::Plot::new("Sun position")
                .view_aspect(1.0)
                .allow_drag(false)
                .allow_zoom(false)
                .allow_scroll(false)
                .allow_boxed_zoom(false)
                .height(250.0)
                .show(ui, |ui| {
                let n = 512;
                let circle_points: egui::plot::PlotPoints = (0..=n)
                    .map(|i| {
                        let t = egui::remap(i as f64, 0.0..=(n as f64), 0.0..=6.28);
                        let r = 1.0;
                        [
                            r * t.cos() + 0.0 as f64,
                            r * t.sin() + 0.0 as f64,
                        ]
                    })
                    .collect();
                ui.line(egui::plot::Line::new(circle_points));

                let sun_pos = [sun_direction.x as f64, sun_direction.z as f64];
                ui.points(egui::plot::Points::new(vec![sun_pos])
                    .color(egui::Color32::GOLD)
                    .shape(egui::plot::MarkerShape::Asterisk)
                    .radius(8.0)
                    .name("Sun position"));
                
                let pointer = ui.pointer_coordinate();
                if let Some(pointer) = pointer {
                    if mouse_down && pointer.x.abs() <= 1.0 && pointer.y.abs() <= 1.0 {
                        let mut new_pos = pointer.to_vec2();
                        if new_pos.length() > 1.0 {
                            new_pos = new_pos.normalized();
                        }
                        let new_pos_y = (1.0 - new_pos.x * new_pos.x - new_pos.y * new_pos.y).sqrt();
                        let new_pos_vec = Vec3::new(new_pos.x as f32, new_pos_y as f32, new_pos.y as f32).normalize();
                        
                        self.tracing_state.config.write().sun_direction = new_pos_vec.extend(sun_direction.w);
                        self.tracing_state.dirty.store(true, Ordering::Relaxed);
                    }
                }
            });
        });
        self.show_environment_window = show_environment_window;
    }

    fn handle_input(&mut self, ui: &egui::Ui) {
        if self.last_input.elapsed().as_millis() < 16 {
            return;
        }
        self.last_input = Instant::now();
    
        if ui.input().pointer.secondary_down() {
            self.tracing_state.interacting.store(true, Ordering::Relaxed);
            self.window.set_cursor_visible(false);
        } else {
            self.tracing_state.interacting.store(false, Ordering::Relaxed);
            self.window.set_cursor_visible(true);
        }
    
        let mut config = self.tracing_state.config.write();
    
        let mut forward = Vec3::new(0.0, 0.0, 1.0);
        let mut right = Vec3::new(1.0, 0.0, 0.0);
        let euler_mat =
            Mat3::from_rotation_y(config.cam_rotation.y) * Mat3::from_rotation_x(config.cam_rotation.x);
        forward = euler_mat * forward;
        right = euler_mat * right;
    
        let speed = if ui.input().modifiers.shift {
            0.5
        } else if ui.input().modifiers.ctrl {
            0.01
        } else {
            0.1
        };

        if ui.input().key_down(egui::Key::W) {
            config.cam_position += forward.extend(0.0) * speed;
        }
        if ui.input().key_down(egui::Key::S) {
            config.cam_position -= forward.extend(0.0) * speed;
        }
        if ui.input().key_down(egui::Key::D) {
            config.cam_position += right.extend(0.0) * speed;
        }
        if ui.input().key_down(egui::Key::A) {
            config.cam_position -= right.extend(0.0) * speed;
        }
        if ui.input().key_down(egui::Key::E) {
            config.cam_position.y += speed;
        }
        if ui.input().key_down(egui::Key::Q) {
            config.cam_position.y -= speed;
        }
    
        config.cam_rotation.x += self.mouse_delta.1 * 0.005;
        config.cam_rotation.y += self.mouse_delta.0 * 0.005;
        self.mouse_delta = (0.0, 0.0);
    }

    pub fn redraw(&mut self, platform: &mut Platform, start_time: &Instant) {
        platform.update_time(start_time.elapsed().as_secs_f64());

        let output_frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(_) => {
                return;
            }
        };
        let output_view = output_frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Begin to draw the UI frame.
        platform.begin_frame();

        // Render here
        egui::CentralPanel::default()
            .frame(egui::Frame::default().inner_margin(egui::Vec2::ZERO))
            .show(&platform.context(), |ui| {
                self.on_gui(&platform.context());
                self.handle_input(ui);

                let rect = ui.allocate_exact_size(ui.available_size(), egui::Sense::drag()).0;
                let framebuffer = self.tracing_state.framebuffer.read().clone(); // TODO: clone is slow
                let width = self.tracing_state.config.read().width;
                let height = self.tracing_state.config.read().height;
                let tonemapping = self.tonemapping;
                let cb = egui_wgpu::CallbackFn::new()
                    .prepare(move |_device, queue, _encoder, typemap| {
                        if let Some(resources) = typemap.get::<PaintCallbackResources>() {
                            resources.prepare(queue, &framebuffer, width, height, tonemapping);
                        }
                        Default::default()
                    })
                    .paint(move |_info, rpass, typemap| {
                        if let Some(resources) = typemap.get::<PaintCallbackResources>() {
                            resources.paint(rpass);
                        }
                    });

                let callback = egui::PaintCallback {
                    rect,
                    callback: Arc::new(cb),
                };

                ui.painter().add(callback);
            });

        // End the UI frame. We could now handle the output and draw the UI with the backend.
        let full_output = platform.end_frame(Some(&self.window));
        let paint_jobs = platform.context().tessellate(full_output.shapes);

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        // Upload all resources for the GPU.
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [self.surface_config.width, self.surface_config.height],
            pixels_per_point: self.window.scale_factor() as f32,
        };
        let tdelta: egui::TexturesDelta = full_output.textures_delta;
        for (id, image_delta) in &tdelta.set {
            self.egui_renderer.update_texture(&self.device, &self.queue, *id, image_delta);
        }
        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &paint_jobs,
            &screen_descriptor,
        );

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &output_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: true,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
            });
            self.egui_renderer.render(&mut render_pass, &paint_jobs, &screen_descriptor);
        }
        // Submit the commands.
        self.queue.submit(iter::once(encoder.finish()));

        // Redraw egui
        output_frame.present();

        for id in &tdelta.free {
            self.egui_renderer.free_texture(id);
        }
    }

    pub fn handle_mouse_motion(&mut self, delta: (f64, f64)) {
        if self.tracing_state.interacting.load(Ordering::Relaxed) {
            self.mouse_delta.0 += delta.0 as f32;
            self.mouse_delta.1 += delta.1 as f32;

            // Set mouse position to center of screen
            let size = self.window.inner_size();
            let center = winit::dpi::LogicalPosition::new(
                size.width as f32 / 2.0,
                size.height as f32 / 2.0,
            );
            let _ = self.window.set_cursor_position(center);
        }
    }

    pub fn handle_resize(&mut self, size: PhysicalSize<u32>) {
        if size.width > 0 && size.height > 0 {
            self.surface_config.width = size.width;
            self.surface_config.height = size.height;
            self.surface.configure(&self.device, &self.surface_config);
        }
    }

    pub fn handle_file_dropped(&mut self, path: &std::path::Path) {
        let path_str = path.to_str().expect("Path was not valid utf8.");
        if is_image(path_str) {
            self.set_skybox(path_str);
        } else {
            self.set_scene(path_str);
        }
    }
}

struct PaintCallbackResources {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    render_buffer: wgpu::Buffer,
}

impl PaintCallbackResources {
    fn prepare(
        &self,
        queue: &wgpu::Queue,
        framebuffer: &Vec<f32>,
        width: u32,
        height: u32,
        tonemapping: Tonemapping,
    ) {
        queue.write_buffer(&self.render_buffer, 0, bytemuck::cast_slice(framebuffer));
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[width, height, tonemapping as u32]),
        );
    }

    fn paint<'rpass>(&'rpass self, rpass: &mut wgpu::RenderPass<'rpass>) {
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.draw(0..6, 0..1);
    }

    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> PaintCallbackResources {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("resources/render.wgsl").into()),
        });
    
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
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
            contents: bytemuck::cast_slice(&[0.0, 0.0, 0.0]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        });
    
        let render_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (width * height * 3 * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
    
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: render_buffer.as_entire_binding(),
                },
            ],
        });
    
        PaintCallbackResources {
            pipeline,
            bind_group,
            uniform_buffer,
            render_buffer,
        }
    }

    fn save_render(&self, texture_width: u32, texture_height: u32, format: wgpu::TextureFormat, device: &wgpu::Device, queue: &wgpu::Queue) {
        let texture_desc = &wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: texture_width,
                height: texture_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        let texture = device.create_texture(texture_desc);
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: true,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
            });
            self.paint(&mut render_pass);
        }
    
        let u32_size = std::mem::size_of::<u32>() as u32;
    
        let output_buffer_size = (u32_size * texture_width * texture_height) as wgpu::BufferAddress;
        let output_buffer_desc = wgpu::BufferDescriptor {
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST
                // this tells wpgu that we want to read this buffer from the cpu
                | wgpu::BufferUsages::MAP_READ,
            label: None,
            mapped_at_creation: false,
        };
        let output_buffer = device.create_buffer(&output_buffer_desc);
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                        texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: NonZeroU32::new(u32_size * texture_width),
                    rows_per_image: NonZeroU32::new(texture_height),
                },
            },
            texture_desc.size,
        );
        queue.submit(Some(encoder.finish()));
    
        {
            let buffer_slice = output_buffer.slice(..);
        
            buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);
            let mut data = buffer_slice.get_mapped_range().to_vec();
            data.chunks_exact_mut(4).for_each(|c| c.swap(0, 2)); // BGRA -> RGBA swizzle
            let Some(buffer) = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(texture_width, texture_height, data) else {
                return;
            };
            let image = image::DynamicImage::ImageRgba8(buffer).into_rgba8();
            if let Some(path) = tinyfiledialogs::save_file_dialog("Save render", "") {
                let res = image.save(path);
                if res.is_err() {
                    #[cfg(debug_assertions)] println!("Failed to save image: {:?}", res.err());
                }
            }
        }
        output_buffer.unmap();
    }
}