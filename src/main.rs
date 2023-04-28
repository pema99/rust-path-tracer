#![feature(int_roundings)]

pub mod window;
use window::open_window;

pub mod trace;

pub mod bvh;

fn main() {
    open_window();
}
