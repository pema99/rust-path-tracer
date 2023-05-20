use glam::{UVec4, Vec4, Mat4, Vec2, Vec3};
use gpgpu::{GpuBuffer, BufOps, GpuConstImage, primitives::pixels::{Rgba8UintNorm}, ImgOps};
use image::DynamicImage;
use russimp::{scene::{Scene, PostProcess::*}, node::Node, material::{DataContent, TextureType, Texture, Material, PropertyTypeInfo}};
use shared_structs::{MaterialData, PerVertexData, LightPickEntry};

use crate::{bvh::{BVH, BVHBuilder}, trace::FW, light_pick};

pub struct World<'fw> {
    pub per_vertex_buffer: GpuBuffer<'fw, PerVertexData>,
    pub index_buffer: GpuBuffer<'fw, UVec4>,
    pub bvh: BVH<'fw>,

    pub atlas: GpuConstImage<'fw, Rgba8UintNorm>,
    pub material_data_buffer: GpuBuffer<'fw, MaterialData>,    
    
    pub light_pick_buffer: GpuBuffer<'fw, LightPickEntry>,
}

fn convert_texture(texture: &Texture) -> Option<DynamicImage> {
    let image = match &texture.data {
        DataContent::Texel(raw_data) => {
            let image_data = raw_data.iter().flat_map(|c| [c.r, c.g, c.b, c.a]).collect::<Vec<_>>();
            let image_buffer = image::RgbaImage::from_vec(texture.width, texture.height, image_data)?;
            image::DynamicImage::ImageRgba8(image_buffer)
        },
        DataContent::Bytes(bytes) => {
            image::io::Reader::new(std::io::Cursor::new(bytes)).with_guessed_format().ok()?.decode().ok()?
        }
    };

    Some(image)
}

fn load_texture(material: &Material, texture_type: TextureType) -> Option<DynamicImage> {
    material.textures.get(&texture_type).and_then(|texture| convert_texture(&texture.borrow()))
}

fn load_float_array(material: &Material, name: &str) -> Option<Vec<f32>> {
    let prop = material.properties.iter().find(|p| p.key == name)?;
    match &prop.data {
        PropertyTypeInfo::FloatArray(col) => Some(col.clone()),
        _ => None
    }
}

impl<'fw> World<'fw> {
    pub fn from_path(path: &str) -> Self {
        let blend = Scene::from_file(
            path,
            vec![
                JoinIdenticalVertices,
                Triangulate,
                SortByPrimitiveType,
                GenerateSmoothNormals,
                GenerateUVCoords,
                TransformUVCoords,
                CalculateTangentSpace,
                EmbedTextures,
            ],
        )
        .unwrap();

        // Gather mesh data
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut normals = Vec::new();
        let mut tangents = Vec::new();
        let mut uvs = Vec::new();

        fn walk_node_graph(
            scene: &Scene,
            node: &Node,
            trs: Mat4,
            vertices: &mut Vec<Vec4>,
            indices: &mut Vec<UVec4>,
            normals: &mut Vec<Vec4>,
            tangents: &mut Vec<Vec4>,
            uvs: &mut Vec<Vec2>
        ) {
            let node_trs = Mat4::from_cols_array_2d(&[
                [node.transformation.a1, node.transformation.b1, node.transformation.c1, node.transformation.d1],
                [node.transformation.a2, node.transformation.b2, node.transformation.c2, node.transformation.d2],
                [node.transformation.a3, node.transformation.b3, node.transformation.c3, node.transformation.d3],
                [node.transformation.a4, node.transformation.b4, node.transformation.c4, node.transformation.d4],
            ]);
            let new_trs = trs * node_trs;
            let (node_scale,node_quat,_) = new_trs.to_scale_rotation_translation();

            for mesh_idx in node.meshes.iter() {
                let mesh = &scene.meshes[*mesh_idx as usize];
                let triangle_offset = vertices.len() as u32;
                for v in &mesh.vertices {
                    let vert = new_trs.mul_vec4(Vec4::new(v.x, v.y, v.z, 1.0));
                    vertices.push(Vec4::new(vert.x, vert.z, vert.y, 1.0));
                }
                for f in &mesh.faces {
                    assert_eq!(f.0.len(), 3);
                    indices.push(UVec4::new(triangle_offset + f.0[0], triangle_offset + f.0[2], triangle_offset + f.0[1], mesh.material_index));
                }
                for n in &mesh.normals {
                    let norm = (node_quat.mul_vec3(Vec3::new(n.x, n.y, n.z) / node_scale)).normalize();
                    normals.push(Vec4::new(norm.x, norm.z, norm.y, 0.0));
                }
                for t in &mesh.tangents {
                    let tan = (node_quat.mul_vec3(Vec3::new(t.x, t.y, t.z) / node_scale)).normalize();
                    tangents.push(Vec4::new(tan.x, tan.z, tan.y, 0.0));
                }
                if let Some(Some(uv_set)) = mesh.texture_coords.iter().next() {
                    for uv in uv_set {
                        uvs.push(Vec2::new(uv.x, uv.y));
                    }
                } else {
                    uvs.resize(vertices.len(), Vec2::ZERO);
                }
            }

            for child in node.children.borrow().iter() {
                walk_node_graph(scene, &child, new_trs, vertices, indices, normals, tangents, uvs);
            }
        }

        if let Some(root) = blend.root.as_ref() {
            walk_node_graph(&blend, root, Mat4::IDENTITY, &mut vertices, &mut indices, &mut normals, &mut tangents, &mut uvs);
        }

        // Gather material data
        let mut material_datas = vec![MaterialData::default(); blend.materials.len()];

        let mut textures = Vec::new();
        for (material_index, material) in blend.materials.iter().enumerate() {
            let current_material_data = &mut material_datas[material_index];
            if let Some(texture) = load_texture(&material, TextureType::Diffuse) {
                // Albedo data is stored in gamma space, but we atlas it with all the other textures
                // which are stored in linear. Therefore, we convert here.
                let mut texture = texture.into_rgb8();
                for pixel in texture.iter_mut() {
                    *pixel = ((*pixel as f32 / 255.0).powf(2.2) * 255.0) as u8;
                }
                textures.push(image::DynamicImage::ImageRgb8(texture));
                current_material_data.set_has_albedo_texture(true);
            }
            if let Some(texture) = load_texture(&material, TextureType::Metalness) {
                textures.push(texture);
                current_material_data.set_has_metallic_texture(true);
            }
            if let Some(texture) = load_texture(&material, TextureType::Roughness) {
                textures.push(texture);
                current_material_data.set_has_roughness_texture(true);
            }
            if let Some(texture) = load_texture(&material, TextureType::Normals) {
                textures.push(texture);
                current_material_data.set_has_normal_texture(true);
            }
            if let Some(col) = load_float_array(&material, "$clr.diffuse") {
                current_material_data.albedo = Vec4::new(col[0], col[1], col[2], col[3]);
            }
            if let Some(col) = load_float_array(&material, "$clr.emissive") {
                // HACK: Multiply by 15 since assimp 5.2.5 doesn't support emissive strength :(
                current_material_data.emissive = Vec4::new(col[0], col[1], col[2], col[3]) * 15.0;
            }
            if let Some(col) = load_float_array(&material, "$mat.metallicFactor") {
                current_material_data.metallic = Vec4::splat(col[0]);
            }
            if let Some(col) = load_float_array(&material, "$mat.roughnessFactor") {
                current_material_data.roughness = Vec4::splat(col[0]);
            }
        }

        let (atlas_raw, mut sts) = crate::atlas::pack_textures(&textures, 4096, 4096);
        let atlas = GpuConstImage::from_bytes(&FW, &atlas_raw.to_rgba8(), 4096, 4096);

        for material_data in material_datas.iter_mut() {
            if material_data.has_albedo_texture() {
                material_data.albedo = sts.remove(0); // TODO: Optimize this
            }
            if material_data.has_metallic_texture() {
                material_data.metallic = sts.remove(0);
            }
            if material_data.has_roughness_texture() {
                material_data.roughness = sts.remove(0);
            }
            if material_data.has_normal_texture() {
                material_data.normals = sts.remove(0);
            }
        }
        let material_data_buffer = GpuBuffer::from_slice(&FW, &material_datas);

        // BVH building
        let now = std::time::Instant::now();
        let bvh = BVHBuilder::new(&vertices, &mut indices).sah_samples(128).build();
        println!("BVH build time: {:?}", now.elapsed());

        // Build light pick table
        let now = std::time::Instant::now();
        let emissive_mask = light_pick::compute_emissive_mask(&indices, &material_datas);
        let light_pick_table = light_pick::build_light_pick_table(&vertices, &indices, &emissive_mask);
        let light_pick_buffer = GpuBuffer::from_slice(&FW, &light_pick_table);
        println!("Light pick table build time: {:?}", now.elapsed());

        // TODO: Seperate loading from GPU upload
        // Upload to GPU
        let mut per_vertex_data = Vec::new();
        for i in 0..vertices.len() {
            per_vertex_data.push(PerVertexData {
                vertex: *vertices.get(i).unwrap_or(&Vec4::ZERO),
                normal: *normals.get(i).unwrap_or(&Vec4::ZERO),
                tangent: *tangents.get(i).unwrap_or(&Vec4::ZERO),
                uv0: *uvs.get(i).unwrap_or(&Vec2::ZERO),
                ..Default::default()
            });
        }
        let per_vertex_buffer = GpuBuffer::from_slice(&FW, &per_vertex_data);
        let index_buffer = GpuBuffer::from_slice(&FW, &indices);
        Self {
            per_vertex_buffer,
            index_buffer,
            bvh,
            atlas,
            material_data_buffer,
            light_pick_buffer,
        }
    }
}