use std::time::Instant;
use egui::FontDefinitions;
use egui_winit_platform::{Platform, PlatformDescriptor};
use winit::event::Event::{DeviceEvent, WindowEvent, MainEventsCleared, RedrawRequested};
use rustic::app::App;
use winit::event_loop::ControlFlow;

fn main() {
    let width = 1280;
    let height = 720;

    let event_loop = winit::event_loop::EventLoopBuilder::<()>::with_user_event().build();
    let window = winit::window::WindowBuilder::new()
        .with_decorations(true)
        .with_resizable(true)
        .with_transparent(false)
        .with_title("rust-path-tracer")
        .with_inner_size(winit::dpi::PhysicalSize {
            width,
            height,
        })
        .build(&event_loop)
        .expect("Building window failed");

    let mut platform = Platform::new(PlatformDescriptor {
        physical_width: width,
        physical_height: height,
        scale_factor: window.scale_factor(),
        font_definitions: FontDefinitions::default(),
        style: Default::default(),
    });

    let mut app = App::new(window);

    let start_time = Instant::now();
    event_loop.run(move |event, _, control_flow| {
        // Pass the winit events to the platform integration.
        platform.handle_event(&event);

        match event {
            RedrawRequested(..) => {
                app.redraw(&mut platform, &start_time);
            }
            DeviceEvent {
                event: winit::event::DeviceEvent::MouseMotion { delta },
                ..
            } => {
                app.handle_mouse_motion(delta);
            }
            MainEventsCleared => {
                app.window().request_redraw();
            }
            WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::Resized(size) => {
                    app.handle_resize(size);
                }
                winit::event::WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                winit::event::WindowEvent::DroppedFile(path) => {
                    app.handle_file_dropped(&path);
                }
                _ => {}
            },
            _ => (),
        }
    });
}