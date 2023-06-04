use std::sync::Arc;

use rustic::trace::*;
use shared_structs::NextEventEstimation;

fn trace(use_cpu: bool, scene: &str, skybox: Option<&str>, state: &Arc<TracingState>) {
    if use_cpu {
        trace_cpu(scene, skybox, state.clone());
    } else {
        trace_gpu(scene, skybox, state.clone());
    }
}

fn furnace_test(use_cpu: bool, use_mis: bool) {
    let size = 128;
    let coord = (65, 75);
    let albedo = 0.8;
    let tolerance = 0.02;

    let state = setup_trace(size as u32, size as u32, 32);
    if use_mis {
        state.config.write().nee = NextEventEstimation::MultipleImportanceSampling.to_u32();
    }
    trace(use_cpu, "scenes/FurnaceTest.glb", None, &state);
    let frame = state.framebuffer.read();

    let pixel_r = frame[(size * 3) * coord.1 + coord.0 * 3 + 0].powf(1.0 / 2.2);
    let pixel_g = frame[(size * 3) * coord.1 + coord.0 * 3 + 1].powf(1.0 / 2.2);
    let pixel_b = frame[(size * 3) * coord.1 + coord.0 * 3 + 2].powf(1.0 / 2.2);
    assert!((pixel_r - albedo).abs() < tolerance);
    assert!((pixel_g - albedo).abs() < tolerance);
    assert!((pixel_b - albedo).abs() < tolerance);
}

#[test]
fn furnace_test_cpu() {
    furnace_test(true, false);
}

#[test]
fn furnace_test_gpu() {
    furnace_test(false, false);
}

#[test]
fn furnace_test_cpu_mis() {
    furnace_test(true, true);
}

#[test]
fn furnace_test_gpu_mis() {
    furnace_test(false, true);
}