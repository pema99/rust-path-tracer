use egui_glium::egui_winit::egui;
use glium::{
    glutin::{
        event::WindowEvent,
        self,
    },
    index::NoIndices,
    texture::{buffer_texture::{BufferTexture, BufferTextureType}},
    uniform, Display, Surface, VertexBuffer,
};
use std::{sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc, Mutex,
}};

use crate::trace::{trace, TracingConfig};

struct AppState {
    rendertarget: BufferTexture<f32>,
    framebuffer: Arc<Mutex<Vec<f32>>>,
    running: Arc<AtomicBool>,
    samples: Arc<AtomicU32>,
    denoise: Arc<AtomicBool>,
    sync_rate: Arc<AtomicU32>,
    config: TracingConfig,
}

fn make_view_dependent_state(
    display: &Display,
    config: Option<TracingConfig>,
) -> (TracingConfig, Arc<Mutex<Vec<f32>>>, BufferTexture<f32>) {
    let inner_size = display.get_framebuffer_dimensions();
    let config = TracingConfig {
        width: inner_size.0,
        height: inner_size.1,
        ..config.unwrap_or_default()
    };
    let data_size = config.width as usize * config.height as usize * 3;
    let framebuffer = Arc::new(Mutex::new(vec![
        0.0;
        data_size
    ]));
    let rendertarget = BufferTexture::empty(display, data_size, BufferTextureType::Float).unwrap();
    (config, framebuffer, rendertarget)
}

impl AppState {
    fn new(display: &Display) -> Self {
        let (config, framebuffer, rendertarget) = make_view_dependent_state(display, None);
        let running = Arc::new(AtomicBool::new(false));
        let samples = Arc::new(AtomicU32::new(0));
        let denoise = Arc::new(AtomicBool::new(false));
        let sync_rate = Arc::new(AtomicU32::new(128));
        Self {
            rendertarget,
            framebuffer,
            running,
            samples,
            denoise,
            sync_rate,
            config,
        }
    }

    fn initialize_for_new_render(&mut self, display: &Display) {
        let (config, framebuffer, rendertarget) =
            make_view_dependent_state(display, Some(self.config));
        self.config = config;
        self.framebuffer = framebuffer;
        self.rendertarget = rendertarget;
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
    let config = app_state.config.clone();
    std::thread::spawn(move || {
        trace(data, running, samples, denoise, sync_rate, config);
    });
}

fn on_gui(display: &Display, app_state: &mut AppState, egui_ctx: &egui::Context) {
    egui::Window::new("Settings").show(egui_ctx, |ui| {
        if app_state.running.load(Ordering::Relaxed) {
            if ui.button("Stop").clicked() {
                display.gl_window().window().set_resizable(true);
                app_state.running.store(false, Ordering::Relaxed);
            }
        } else {
            if ui.button("Start").clicked() {
                display.gl_window().window().set_resizable(false);
                app_state.initialize_for_new_render(&display);
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
            if ui.add(egui::Slider::new(&mut sync_rate, 1..=1024).text("Sync rate")).changed() {
                app_state.sync_rate.store(sync_rate, Ordering::Relaxed);
            }
        }

        ui.label(format!(
            "Samples: {}",
            app_state.samples.load(Ordering::Relaxed)
        ));
    });
}

pub fn open_window() {
    let event_loop = glutin::event_loop::EventLoopBuilder::with_user_event().build();
    let display = create_display(&event_loop);

    let mut egui_glium = egui_glium::EguiGlium::new(&display, &event_loop);

    let mut app_state = AppState::new(&display);
    let (vertex_buffer, index_buffer, program) = setup_program(&display);

    // Used to only redraw when something has changed
    let mut handled_samples = 0;

    event_loop.run(move |event, _, control_flow| {
        // If samples have gone out of sync, force redraw
        let current_samples = app_state.samples.load(Ordering::Relaxed);
        if handled_samples != current_samples {
            handled_samples = current_samples;
            display.gl_window().window().request_redraw();
        }

        let mut redraw = || {
            let repaint_after = egui_glium.run(&display, |egui_ctx| {
                on_gui(&display, &mut app_state, egui_ctx);
            });

            if repaint_after.is_zero() {
                display.gl_window().window().request_redraw();
                glutin::event_loop::ControlFlow::Poll
            } else if let Some(repaint_after_instant) = std::time::Instant::now().checked_add(repaint_after) {
                glutin::event_loop::ControlFlow::WaitUntil(repaint_after_instant)
            } else {
                glutin::event_loop::ControlFlow::Wait
            };

            {
                let mut target = display.draw();

                // Readback buffer from compute thread and render it
                {
                    let framebuffer_data = match app_state.framebuffer.lock() {
                        Ok(fb) => fb.to_owned(),
                        Err(_) => panic!(),
                    };
                    app_state.rendertarget.write(&framebuffer_data);
                    target
                        .draw(
                            &vertex_buffer,
                            &index_buffer,
                            &program,
                            &uniform! { 
                                texture: &app_state.rendertarget,
                                width: app_state.config.width as f32,
                                height: app_state.config.height as f32,
                            },
                            &Default::default(),
                        )
                        .unwrap();
                }

                egui_glium.paint(&display, &mut target);
                target.finish().unwrap();
            }
        };

        match event {
            glutin::event::Event::RedrawRequested(_) => redraw(),
            glutin::event::Event::WindowEvent { event, .. } => {
                if matches!(event, WindowEvent::CloseRequested | WindowEvent::Destroyed) {
                    *control_flow = glutin::event_loop::ControlFlow::Exit;
                }

                let event_response = egui_glium.on_event(&event);

                if event_response.repaint {
                    display.gl_window().window().request_redraw();
                }
            }
            glutin::event::Event::NewEvents(glutin::event::StartCause::ResumeTimeReached {
                ..
            }) => {
                display.gl_window().window().request_redraw();
            }
            _ => (),
        }
    });
}

fn create_display(event_loop: &glutin::event_loop::EventLoop<()>) -> glium::Display {
    let window_builder = glutin::window::WindowBuilder::new()
        .with_resizable(true)
        .with_inner_size(glutin::dpi::LogicalSize {
            width: 1280.0,
            height: 720.0,
        })
        .with_title("rustic");

    let context_builder = glutin::ContextBuilder::new()
        .with_depth_buffer(0)
        .with_stencil_buffer(0)
        .with_vsync(true);

    glium::Display::new(window_builder, context_builder, event_loop).unwrap()
}

// Setup vert buffer for a quad
#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 2],
}
glium::implement_vertex!(Vertex, position);

fn setup_program(display: &Display) -> (VertexBuffer<Vertex>, NoIndices, glium::Program) {
    let vertex1 = Vertex {
        position: [-1.0, -1.0],
    };
    let vertex2 = Vertex {
        position: [-1.0, 1.0],
    };
    let vertex3 = Vertex {
        position: [1.0, -1.0],
    };
    let vertex4 = Vertex {
        position: [1.0, 1.0],
    };
    let shape = vec![vertex1, vertex2, vertex3, vertex4, vertex2, vertex3];

    let vertex_buffer = VertexBuffer::new(display, &shape).unwrap();
    let indices = NoIndices(glium::index::PrimitiveType::TrianglesList);

    let code_vert = include_str!("shaders/basic_vert.glsl");
    let code_frag = include_str!("shaders/basic_frag.glsl");
    let program = glium::Program::from_source(display, &code_vert, &code_frag, None).unwrap();

    (vertex_buffer, indices, program)
}