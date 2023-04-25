use spirv_std::glam::{UVec2, Vec2};

pub fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u32 + 2891336453u32;
    let word = ((state >> ((state >> 28u32) + 4u32)) ^ state) * 277803737u32;
    (word >> 22u32) ^ word
}

pub struct RngState<'a>(&'a mut UVec2);

impl<'a> RngState<'a> {
    pub fn new(seed: &'a mut UVec2) -> Self {
        Self(seed)
    }

    pub fn gen_float_pair(&mut self) -> Vec2 {
        self.0.x = pcg_hash(self.0.x);
        self.0.y = pcg_hash(self.0.y);
        Vec2::new(
            self.0.x as f32 / u32::MAX as f32,
            self.0.y as f32 / u32::MAX as f32,
        )
    }

    pub fn gen_float(&mut self) -> f32 {
        self.gen_float_pair().x
    }
}
