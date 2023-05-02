use image::{DynamicImage, GenericImage};

#[derive(Clone, Copy)]
pub struct PackingRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

struct PackingTreeNode {
    children: Option<[Box<PackingTreeNode>; 4]>,
    rect: PackingRect,
}

impl PackingTreeNode {
    fn new(rect: PackingRect) -> Self {
        Self {
            children: None,
            rect,
        }
    }

    fn subdivide(&mut self) {
        if self.children.is_some() {
            return;
        }

        let rect = self.rect;
        let half_width = rect.width / 2;
        let half_height = rect.height / 2;
        let children = [
            Box::new(PackingTreeNode::new(PackingRect {
                x: rect.x,
                y: rect.y,
                width: half_width,
                height: half_height,
            })),
            Box::new(PackingTreeNode::new(PackingRect {
                x: rect.x + half_width,
                y: rect.y,
                width: half_width,
                height: half_height,
            })),
            Box::new(PackingTreeNode::new(PackingRect {
                x: rect.x,
                y: rect.y + half_height,
                width: half_width,
                height: half_height,
            })),
            Box::new(PackingTreeNode::new(PackingRect {
                x: rect.x + half_width,
                y: rect.y + half_height,
                width: half_width,
                height: half_height,
            })),
        ];
        self.children = Some(children);
    }

    fn find_first_leaf(&mut self) -> Option<&mut PackingTreeNode> {
        if self.children.is_none() {
            return Some(self);
        }

        let children = self.children.as_mut().unwrap();
        if children.iter().any(|child| child.children.is_none()) {
            for child in children.iter_mut() {
                if child.children.is_none() {
                    return Some(child.as_mut());
                }
            }
        } else {
            for child in children.iter_mut() {
                if let Some(leaf) = child.find_first_leaf() {
                    return Some(leaf);
                }
            }
        }

        None
    }

    fn count_leafs(&self) -> usize {
        if self.children.is_none() {
            return 1;
        }

        let mut count = 0;
        for child in self.children.as_ref().unwrap().iter() {
            count += child.count_leafs();
        }
        count
    }

    fn gather_leafs(&self, rects: &mut Vec<PackingRect>) {
        if self.children.is_none() {
            rects.push(self.rect);
            return;
        }

        for child in self.children.as_ref().unwrap().iter() {
            child.gather_leafs(rects);
        }
    }
}

pub fn pack_textures(textures: &[DynamicImage], atlas_width: u32, atlas_height: u32) -> (DynamicImage, Vec<PackingRect>) {
    let mut root = PackingTreeNode::new(PackingRect {
        x: 0,
        y: 0,
        width: atlas_width,
        height: atlas_height,
    });

    // This ain't great performance-wise, but whatever
    while root.count_leafs() < textures.len() {
        match root.find_first_leaf() {
            Some(leaf) => {
                leaf.subdivide();
            }
            None => {
                break;
            }
        }
    }

    let mut leafs = Vec::new();
    root.gather_leafs(&mut leafs);
    leafs.sort_by(|a, b| b.width.cmp(&a.width));
    leafs.truncate(textures.len());

    let mut atlas = DynamicImage::new_rgba8(atlas_width, atlas_height);
    for (i, leaf) in leafs.iter().enumerate() {
        let tex = &textures[i];
        let resized_tex = tex.resize_exact(leaf.width, leaf.height, image::imageops::FilterType::Gaussian);
        atlas.copy_from(&resized_tex, leaf.x, leaf.y).unwrap();
    }

    (atlas, leafs)
}


