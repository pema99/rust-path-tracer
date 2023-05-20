use glam::{UVec4, Vec3, Vec4, Vec4Swizzles};
use rand::Rng;
use shared_structs::{LightPickEntry, MaterialData};

fn triangle_area(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let side_a = b - a;
    let side_b = c - b;
    let side_c = a - c;
    let s = (side_a.length() + side_b.length() + side_c.length()) / 2.0;
    (s * (s - side_a.length()) * (s - side_b.length()) * (s - side_c.length())).sqrt()
}

pub fn compute_emissive_mask(indices: &[UVec4], material_datas: &[MaterialData]) -> Vec<bool> {
    let mut emissive_mask = vec![false; indices.len()];
    for i in 0..indices.len() {
        if material_datas[indices[i].w as usize].emissive.xyz() != Vec3::ZERO {
            emissive_mask[i] = true;
        }
    }
    emissive_mask
}

// NOTE: `mask` indicates which triangles are valid for picking
pub fn build_light_pick_table(
    vertices: &[Vec4],
    indices: &[UVec4],
    mask: &[bool],
    material_datas: &[MaterialData],
) -> Vec<LightPickEntry> {
    // Calculate areas and probabilities of picking each triangle
    let mut triangle_areas = vec![0.0; indices.len()];
    let mut triangle_powers = vec![0.0; indices.len()];
    let mut total_power = 0.0;
    let mut total_tris = 0;
    for i in 0..indices.len() {
        if !mask[i] {
            continue;
        }
        total_tris += 1;

        let triangle = indices[i];
        let a = vertices[triangle.x as usize].xyz();
        let b = vertices[triangle.y as usize].xyz();
        let c = vertices[triangle.z as usize].xyz();

        let triangle_area = triangle_area(a, b, c);
        triangle_areas[i] = triangle_area;

        let triangle_power = material_datas[triangle.w as usize].emissive.xyz().dot(Vec3::ONE) * triangle_area;
        triangle_powers[i] = triangle_power;
        total_power += triangle_power;
    }
    if total_tris == 0 {
        // If there are 0 entries, put in a stupid sentinel value
        return vec![LightPickEntry {
            ratio: -1.0,
            ..Default::default()
        }];
    }
    let mut triangle_probabilities = vec![0.0; indices.len()];
    for i in 0..indices.len() {
        triangle_probabilities[i] = triangle_powers[i] / total_power;
    }
    let average_probability = triangle_probabilities.iter().sum::<f32>() / total_tris as f32;
    // Build histogram bins. Each entry contains 2 discrete outcomes.
    #[derive(Debug)]
    struct TriangleBin {
        index_a: usize,
        probability_a: f32,
        index_b: usize,
        probability_b: f32,
    }
    let mut bins = triangle_probabilities
        .iter()
        .enumerate()
        .map(|x| TriangleBin {
            index_a: x.0,
            probability_a: *x.1,
            index_b: 0,
            probability_b: 0.0,
        })
        .filter(|x| x.probability_a != 0.0)
        .collect::<Vec<_>>();
    bins.sort_by(|a, b| {
        a.probability_a
            .partial_cmp(&b.probability_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Robin hood - take from the most probable and give to the least probable
    let num_bins = bins.len();
    let mut most_probable = num_bins - 1;
    for i in 0..num_bins {
        let needed = average_probability - bins[i].probability_a;
        if needed <= 0.0 {
            break;
        }

        bins[i].index_b = bins[most_probable].index_a;
        bins[i].probability_b = needed;
        bins[most_probable].probability_a -= needed;
        if bins[most_probable].probability_a <= average_probability {
            most_probable -= 1;
        }
    }

    // Build the table
    let table = bins
        .iter()
        .map(|x| LightPickEntry {
            triangle_index_a: x.index_a as u32,
            triangle_index_b: x.index_b as u32,
            triangle_pick_pdf_a: triangle_probabilities[x.index_a],
            triangle_area_a: triangle_areas[x.index_a],
            triangle_area_b: triangle_areas[x.index_b],
            triangle_pick_pdf_b: triangle_probabilities[x.index_b],
            ratio: x.probability_a / (x.probability_a + x.probability_b),
        })
        .collect::<Vec<_>>();

    table
}

// Just for reference
#[allow(dead_code)]
fn pick_light(table: &[LightPickEntry]) -> u32 {
    let rng = rand::thread_rng().gen_range(0..table.len());
    let entry = table[rng];
    let rng = rand::thread_rng().gen_range(0.0..1.0);
    if rng < entry.ratio {
        entry.triangle_index_a
    } else {
        entry.triangle_index_b
    }
}

#[allow(dead_code)]
fn build_light_cdf_table(vertices: &[Vec4], indices: &[UVec4], mask: &[bool]) -> Vec<f32> {
    // Calculate areas and probabilities of picking each triangle
    let mut triangle_areas = vec![0.0; indices.len()];
    let mut total_area = 0.0;
    for i in 0..indices.len() {
        if !mask[i] {
            continue;
        }
        let triangle = indices[i];
        let a = vertices[triangle.x as usize].xyz();
        let b = vertices[triangle.y as usize].xyz();
        let c = vertices[triangle.z as usize].xyz();
        let triangle_area = triangle_area(a, b, c);
        total_area += triangle_area;
        triangle_areas[i] = triangle_area;
    }
    let mut triangle_probabilities = vec![0.0; indices.len()];
    for i in 0..indices.len() {
        triangle_probabilities[i] = triangle_areas[i] / total_area;
    }
    for i in 1..indices.len() {
        triangle_probabilities[i] += triangle_probabilities[i - 1];
    }
    triangle_probabilities
}

/*
pub fn compare_approaches(vertices: &[Vec4], indices: &[UVec4], mask: &[bool]) {
    let table = build_light_pick_table(vertices, indices, mask);
    let cdf_table = build_light_cdf_table(vertices, indices, mask);
    let root = BitMapBackend::new("bla.png", (640, 480)).into_drawing_area();

    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(40)
        .margin(5)
        .caption("Histogram sampling", ("sans-serif", 50.0))
        .build_cartesian_2d((0usize..cdf_table.len()).into_segmented(), 0.0f32..0.02f32)
        .unwrap();

    chart
        .configure_mesh()
        .disable_x_mesh()
        .bold_line_style(&WHITE.mix(0.3))
        .y_desc("Count")
        .x_desc("Bucket")
        .axis_desc_style(("sans-serif", 15))
        .draw()
        .unwrap();

    let samples = 100000;
    let mut data = vec![0; indices.len()];
    for i in 0..samples {
        data[pick_light(&table) as usize] += 1;
    }
    /*for i in 0..samples {
        let rng = rand::thread_rng().gen_range(0.0..1.0);
        let mut j = 0;
        while j <= cdf_table.len() {
            if rng < cdf_table[j] {
                break;
            }
            j += 1;
        }
        data[j] += 1;
    }*/
    let data = data
        .iter()
        .map(|x| *x as f32 / samples as f32)
        .collect::<Vec<_>>();

    chart
        .draw_series(
            Histogram::vertical(&chart)
                .style(RED.mix(0.5).filled())
                .data(data.into_iter().enumerate().collect::<Vec<_>>()),
        )
        .unwrap();

    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");

    let root = BitMapBackend::new("bla2.png", (640, 480)).into_drawing_area();

    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(40)
        .margin(5)
        .caption("CDF sampling", ("sans-serif", 50.0))
        .build_cartesian_2d((0usize..cdf_table.len()).into_segmented(), 0.0f32..0.02f32)
        .unwrap();

    chart
        .configure_mesh()
        .disable_x_mesh()
        .bold_line_style(&WHITE.mix(0.3))
        .y_desc("Count")
        .x_desc("Bucket")
        .axis_desc_style(("sans-serif", 15))
        .draw()
        .unwrap();

    let mut data = vec![0; indices.len()];
    /*for i in 0..samples {
        let rng = rand::thread_rng().gen_range(0..table.len());
        let entry = table[rng];
        let rng = rand::thread_rng().gen_range(0.0..1.0);
        let choice = if rng > entry.2 { entry.0 } else { entry.1 };
        data[choice as usize] += 1;
    }*/
    for i in 0..samples {
        let rng = rand::thread_rng().gen_range(0.0..1.0);
        let mut j = 0;
        while j < cdf_table.len() - 1 && rng > cdf_table[j] {
            j += 1;
        }
        data[j] += 1;
    }
    let data = data
        .iter()
        .map(|x| *x as f32 / samples as f32)
        .collect::<Vec<_>>();

    chart
        .draw_series(
            Histogram::vertical(&chart)
                .style(RED.mix(0.5).filled())
                .data(data.into_iter().enumerate().collect::<Vec<_>>()),
        )
        .unwrap();

    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
}
*/
