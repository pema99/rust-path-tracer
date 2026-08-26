use std::sync::Arc;
use rustic::app::App;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

#[derive(Default)]
struct Runner {
    app: Option<App>,
}

impl ApplicationHandler for Runner {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.app.is_some() {
            return;
        }

        let width = 1280;
        let height = 720;

        let window = event_loop.create_window(Window::default_attributes()
            .with_decorations(true)
            .with_resizable(true)
            .with_transparent(false)
            .with_title("rust-path-tracer")
            .with_inner_size(winit::dpi::PhysicalSize {
                width,
                height,
            }))
            .expect("Building window failed");

        self.app = Some(App::new(Arc::new(window)));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        let Some(app) = self.app.as_mut() else {
            return;
        };

        // Pass the winit events to the platform integration.
        app.handle_window_event(&event);

        match event {
            WindowEvent::RedrawRequested => {
                app.redraw();
            }
            WindowEvent::Resized(size) => {
                app.handle_resize(size);
            }
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::DroppedFile(path) => {
                app.handle_file_dropped(&path);
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _device_id: DeviceId, event: DeviceEvent) {
        if let (Some(app), DeviceEvent::MouseMotion { delta }) = (self.app.as_mut(), event) {
            app.handle_mouse_motion(delta);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(app) = self.app.as_ref() {
            app.window().request_redraw();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut Runner::default()).expect("Event loop failed");
}
