use glium::{
    glutin::{
        dpi::LogicalSize,
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::WindowBuilder,
        ContextBuilder,
    },
    index::NoIndices,
    texture::{MipmapsOption, RawImage2d, UncompressedFloatFormat},
    uniform, Display, Rect, Surface, Texture2d, VertexBuffer,
};
use imgui_winit_support::WinitPlatform;
use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc, Mutex,
};

use crate::trace::{trace, TracingConfig};

struct AppState {
    rendertarget: Texture2d,
    framebuffer: Arc<Mutex<Vec<f32>>>,
    running: Arc<AtomicBool>,
    samples_readback: Arc<AtomicU32>,
    denoise: Arc<AtomicBool>,
    config: TracingConfig,
}

fn make_view_dependent_state(
    display: &Display,
) -> (TracingConfig, Arc<Mutex<Vec<f32>>>, Texture2d) {
    let inner_size = display.get_framebuffer_dimensions();
    let config = TracingConfig {
        width: inner_size.0,
        height: inner_size.1,
    };
    let framebuffer = Arc::new(Mutex::new(vec![
        0.0;
        config.width as usize
            * config.height as usize
            * 3
    ]));
    let rendertarget = Texture2d::empty_with_format(
        display,
        UncompressedFloatFormat::F32F32F32,
        MipmapsOption::NoMipmap,
        config.width,
        config.height,
    )
    .unwrap();
    (config, framebuffer, rendertarget)
}

impl AppState {
    fn new(display: &Display) -> Self {
        let (config, framebuffer, rendertarget) = make_view_dependent_state(display);
        let running = Arc::new(AtomicBool::new(false));
        let denoise = Arc::new(AtomicBool::new(false));
        let samples = Arc::new(AtomicU32::new(0));
        Self {
            rendertarget,
            framebuffer,
            running,
            samples_readback: samples,
            denoise,
            config,
        }
    }

    fn initialize_for_new_render(&mut self, display: &Display) {
        let (config, framebuffer, rendertarget) = make_view_dependent_state(display);
        self.config = config;
        self.framebuffer = framebuffer;
        self.rendertarget = rendertarget;
        self.samples_readback.store(0, Ordering::Relaxed);
    }
}

pub fn open_window() {
    let (event_loop, display) = create_window();
    let (mut winit_platform, mut imgui_context) = imgui_init(&display);

    let mut renderer = imgui_glium_renderer::Renderer::init(&mut imgui_context, &display)
        .expect("Failed to initialize renderer");

    let (vertex_buffer, index_buffer, program) = setup_program(&display);
    let mut app_state = AppState::new(&display);

    let mut last_frame = std::time::Instant::now();
    event_loop.run(move |event, _, control_flow| match event {
        Event::NewEvents(_) => {
            let now = std::time::Instant::now();
            imgui_context.io_mut().update_delta_time(now - last_frame);
            last_frame = now;
        }
        Event::MainEventsCleared => {
            let gl_window = display.gl_window();
            winit_platform
                .prepare_frame(imgui_context.io_mut(), gl_window.window())
                .expect("Failed to prepare frame");
            gl_window.window().request_redraw();
        }
        Event::RedrawRequested(_) => {
            let ui = imgui_context.frame();

            ui.window("Settings")
                .size([200.0, 120.0], imgui::Condition::Appearing)
                .build(|| {
                    if app_state.running.load(Ordering::Relaxed) {
                        if ui.button("Stop") {
                            display.gl_window().window().set_resizable(true);
                            app_state.running.store(false, Ordering::Relaxed);
                        }
                    } else {
                        if ui.button("Start") {
                            display.gl_window().window().set_resizable(false);
                            app_state.initialize_for_new_render(&display);
                            start_render(&app_state);
                        }
                    }

                    #[cfg(feature = "oidn")]
                    {
                        let mut denoise_checked = app_state.denoise.load(Ordering::Relaxed);
                        if ui.checkbox("Denoise", &mut denoise_checked) {
                            app_state.denoise.store(denoise_checked, Ordering::Relaxed);
                        }
                    }

                    ui.label_text(
                        &app_state
                            .samples_readback
                            .load(Ordering::Relaxed)
                            .to_string(),
                        "Current SPP",
                    );
                });

            // Setup for drawing
            let gl_window = display.gl_window();
            let mut target = display.draw();
            target.clear_color_srgb(1.0, 1.0, 1.0, 1.0);

            // Render framebuffer
            match app_state.framebuffer.lock() {
                Ok(fb) => {
                    let glium_image = RawImage2d::from_raw_rgb(
                        fb.to_vec(),
                        (app_state.config.width, app_state.config.height),
                    );
                    app_state.rendertarget.write(
                        Rect {
                            left: 0,
                            bottom: 0,
                            width: app_state.config.width,
                            height: app_state.config.height,
                        },
                        glium_image,
                    );
                }
                Err(_) => {}
            }
            target
                .draw(
                    &vertex_buffer,
                    &index_buffer,
                    &program,
                    &uniform! { texture: &app_state.rendertarget },
                    &Default::default(),
                )
                .unwrap();

            // ImGui rendering
            winit_platform.prepare_render(ui, gl_window.window());
            let draw_data = imgui_context.render();
            renderer
                .render(&mut target, draw_data)
                .expect("Rendering failed");
            target.finish().expect("Failed to swap buffers");
        }
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => *control_flow = ControlFlow::Exit,
        event => {
            let gl_window = display.gl_window();
            winit_platform.handle_event(imgui_context.io_mut(), gl_window.window(), &event);
        }
    });
}

fn start_render(app_state: &AppState) {
    app_state.running.store(true, Ordering::Relaxed);
    let data = app_state.framebuffer.clone();
    let running = app_state.running.clone();
    let samples_readback = app_state.samples_readback.clone();
    let denoise = app_state.denoise.clone();
    let config = app_state.config.clone();
    std::thread::spawn(move || {
        trace(data, running, samples_readback, denoise, config);
    });
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

fn create_window() -> (EventLoop<()>, Display) {
    let event_loop = EventLoop::new();
    let context = ContextBuilder::new().with_vsync(true);
    let builder = WindowBuilder::new()
        .with_title("rust-pt-electric-boogaloo")
        .with_inner_size(LogicalSize::new(1280f64, 720f64));
    let display =
        Display::new(builder, context, &event_loop).expect("Failed to initialize display");

    (event_loop, display)
}

fn imgui_init(display: &Display) -> (WinitPlatform, imgui::Context) {
    let mut imgui_context = imgui::Context::create();
    imgui_context.set_ini_filename(None);

    let mut winit_platform = WinitPlatform::init(&mut imgui_context);

    let gl_window = display.gl_window();
    let window = gl_window.window();

    let dpi_mode = imgui_winit_support::HiDpiMode::Default;

    winit_platform.attach_window(imgui_context.io_mut(), window, dpi_mode);

    imgui_context
        .fonts()
        .add_font(&[imgui::FontSource::DefaultFontData { config: None }]);

    (winit_platform, imgui_context)
}
