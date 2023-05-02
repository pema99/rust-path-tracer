use glam::{UVec4, Vec4, Mat4};
use gpgpu::{GpuBuffer, BufOps};
use image::DynamicImage;
use russimp::{scene::{Scene, PostProcess::*}, node::Node, material::{DataContent, TextureType, Texture, Material}};

use crate::{bvh::{BVH, BVHBuilder}, trace::FW};

pub struct World<'fw> {
    pub vertex_buffer: GpuBuffer<'fw, Vec4>,
    pub index_buffer: GpuBuffer<'fw, UVec4>,
    pub normal_buffer: GpuBuffer<'fw, Vec4>,
    pub bvh: BVH<'fw>,
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
                EmbedTextures,
            ],
        )
        .unwrap();

        // Gather mesh data
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut normals = Vec::new();

        fn walk_node_graph(scene: &Scene, node: &Node, trs: Mat4, vertices: &mut Vec<Vec4>, indices: &mut Vec<UVec4>, normals: &mut Vec<Vec4>) {
            let node_trs = Mat4::from_cols_array_2d(&[
                [node.transformation.a1, node.transformation.b1, node.transformation.c1, node.transformation.d1],
                [node.transformation.a2, node.transformation.b2, node.transformation.c2, node.transformation.d2],
                [node.transformation.a3, node.transformation.b3, node.transformation.c3, node.transformation.d3],
                [node.transformation.a4, node.transformation.b4, node.transformation.c4, node.transformation.d4],
            ]);
            let new_trs = node_trs * trs;

            for mesh_idx in node.meshes.iter() {
                let mesh = &scene.meshes[*mesh_idx as usize];
                let triangle_offset = vertices.len() as u32;
                for v in &mesh.vertices {
                    let vert = new_trs.mul_vec4(Vec4::new(v.x, v.y, v.z, 1.0));
                    vertices.push(Vec4::new(vert.x, vert.z, vert.y, 1.0));
                }
                for f in &mesh.faces {
                    assert_eq!(f.0.len(), 3);
                    indices.push(UVec4::new(triangle_offset + f.0[0], triangle_offset + f.0[2], triangle_offset + f.0[1], *mesh_idx as u32));
                }
                for n in &mesh.normals {
                    let norm = new_trs.mul_vec4(Vec4::new(n.x, n.y, n.z, 0.0)).normalize();
                    normals.push(Vec4::new(norm.x, norm.z, norm.y, 0.0));
                }
            }

            for child in node.children.borrow().iter() {
                walk_node_graph(scene, &child, new_trs, vertices, indices, normals);
            }
        }

        if let Some(root) = blend.root.as_ref() {
            walk_node_graph(&blend, root, Mat4::IDENTITY, &mut vertices, &mut indices, &mut normals);
        }

        // Gather material data
        let mut albedo_textures = Vec::new();
        for material in &blend.materials {
            if let Some(texture) = load_texture(&material, TextureType::Diffuse) {
                albedo_textures.push(texture);
            }
        }

        let (atlas, sts) = crate::atlas::pack_textures(&albedo_textures, 1024, 1024);
        atlas.save("atlas.png").unwrap();

        // BVH building
        let now = std::time::Instant::now();
        let bvh = BVHBuilder::new(&vertices, &indices).sah_samples(128).build();
        println!("BVH build time: {:?}", now.elapsed());
        println!("BVH node count: {}", bvh.nodes.len());

        // TODO: Seperate loading from GPU upload
        // Upload to GPU
        let vertex_buffer = GpuBuffer::from_slice(&FW, &vertices);
        let index_buffer = GpuBuffer::from_slice(&FW, &indices);
        let normal_buffer = GpuBuffer::from_slice(&FW, &normals);
        Self {
            vertex_buffer,
            index_buffer,
            normal_buffer,
            bvh,
        }
    }
}