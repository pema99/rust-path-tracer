use std::sync::{Mutex, Arc};
use std::sync::atomic::AtomicBool;
use glium::glutin::event::{Event, WindowEvent};
use glium::glutin::event_loop::{ControlFlow, EventLoop};
use glium::Surface;

use crate::trace::trace;

const TITLE: &str = "rust-pt-electric-boogaloo";

pub fn open_window() {
    let (event_loop, display) = create_window();
    let (mut winit_platform, mut imgui_context) = imgui_init(&display);

    let mut renderer = imgui_glium_renderer::Renderer::init(&mut imgui_context, &display)
        .expect("Failed to initialize renderer");

    let (vertex_buffer, index_buffer, program, texture) = setup_framebuffer(&display);

    let (framebuffer, running) = start_render();

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
            // Create frame for the all important `&imgui::Ui`
            let ui = imgui_context.frame();

            // Draw our example content
            ui.show_demo_window(&mut true);

            // Setup for drawing
            let gl_window = display.gl_window();
            let mut target = display.draw();

            // Renderer doesn't automatically clear window
            target.clear_color_srgb(1.0, 1.0, 1.0, 1.0);

            match framebuffer.lock() {
                Ok(fb) => {
                    let glium_image = glium::texture::RawImage2d::from_raw_rgb(fb.to_vec(), (1280, 720));
                    texture.write(glium::Rect { left: 0, bottom: 0, width: 1280, height: 720 }, glium_image);
                }
                Err(_) => {}
            }
            target.draw(&vertex_buffer, &index_buffer, &program, &glium::uniform! { texture: &texture }, &Default::default()).unwrap();

            // Perform rendering
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

fn start_render() -> (Arc<Mutex<Vec<f32>>>, Arc<AtomicBool>) {
    let data = Arc::new(Mutex::new(vec![0.0; 1280*720*3]));
    let running = Arc::new(AtomicBool::new(true));
    {
        let data = data.clone();
        let running = running.clone();
        std::thread::spawn(move || {
            trace(data, running);
        });
    }
    (data, running)
}

// Setup vert buffer for a quad
#[derive(Copy, Clone)]
struct Vertex { position: [f32; 2] }
glium::implement_vertex!(Vertex, position);

fn setup_framebuffer(display: &glium::Display) -> (glium::VertexBuffer<Vertex>, glium::index::NoIndices, glium::Program, glium::Texture2d) {
    let vertex1 = Vertex { position: [-1.0, -1.0] };
    let vertex2 = Vertex { position: [-1.0,  1.0] };
    let vertex3 = Vertex { position: [ 1.0, -1.0] };
    let vertex4 = Vertex { position: [ 1.0,  1.0] };
    let shape = vec![vertex1, vertex2, vertex3, vertex4, vertex2, vertex3];

    let vertex_buffer = glium::VertexBuffer::new(display, &shape).unwrap();
    let indices = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);

    let code_vert = include_str!("shaders/basic_vert.glsl");
    let code_frag = include_str!("shaders/basic_frag.glsl");
    let program = glium::Program::from_source(display, &code_vert, &code_frag, None).unwrap();

    let framebuffer = glium::Texture2d::empty_with_format(display,
        glium::texture::UncompressedFloatFormat::F32F32F32,
        glium::texture::MipmapsOption::NoMipmap,
        display.gl_window().window().inner_size().width,
        display.gl_window().window().inner_size().height).unwrap();

    (vertex_buffer, indices, program, framebuffer)
}

fn create_window() -> (EventLoop<()>, glium::Display) {
    let event_loop = EventLoop::new();
    let context = glium::glutin::ContextBuilder::new().with_vsync(true);
    let builder = glium::glutin::window::WindowBuilder::new()
        .with_title(TITLE.to_owned())
        .with_inner_size(glium::glutin::dpi::LogicalSize::new(1280f64, 720f64));
    let display =
        glium::Display::new(builder, context, &event_loop).expect("Failed to initialize display");

    (event_loop, display)
}

fn imgui_init(display: &glium::Display) -> (imgui_winit_support::WinitPlatform, imgui::Context) {
    let mut imgui_context = imgui::Context::create();
    imgui_context.set_ini_filename(None);

    let mut winit_platform = imgui_winit_support::WinitPlatform::init(&mut imgui_context);

    let gl_window = display.gl_window();
    let window = gl_window.window();

    let dpi_mode = imgui_winit_support::HiDpiMode::Default;

    winit_platform.attach_window(imgui_context.io_mut(), window, dpi_mode);

    imgui_context
        .fonts()
        .add_font(&[imgui::FontSource::DefaultFontData { config: None }]);

    (winit_platform, imgui_context)
}