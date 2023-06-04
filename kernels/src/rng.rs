use spirv_std::glam::{UVec2, Vec2, Vec3};

#[allow(dead_code)]
#[cfg(target_arch = "spirv")]
pub fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u32 + 2891336453u32;
    let word = ((state >> ((state >> 28u32) + 4u32)) ^ state) * 277803737u32;
    (word >> 22u32) ^ word
}

#[allow(dead_code)]
#[cfg(not(target_arch = "spirv"))]
pub fn pcg_hash(input: u32) -> u32 {
    let state = input.overflowing_mul(747796405u32).0.overflowing_add(2891336453u32).0;
    let word = ((state >> ((state >> 28u32) + 4u32)) ^ state).overflowing_mul(277803737u32).0;
    (word >> 22u32) ^ word
}

// From loicvdbruh: https://www.shadertoy.com/view/NlGXzz. Square roots of primes.
const LDS_MAX_DIMENSIONS: usize = 32;
const LDS_PRIMES: [u32; LDS_MAX_DIMENSIONS] = [
    0x6a09e667u32, 0xbb67ae84u32, 0x3c6ef372u32, 0xa54ff539u32, 0x510e527fu32, 0x9b05688au32, 0x1f83d9abu32, 0x5be0cd18u32,
    0xcbbb9d5cu32, 0x629a2929u32, 0x91590159u32, 0x452fecd8u32, 0x67332667u32, 0x8eb44a86u32, 0xdb0c2e0bu32, 0x47b5481du32,
    0xae5f9155u32, 0xcf6c85d1u32, 0x2f73477du32, 0x6d1826cau32, 0x8b43d455u32, 0xe360b595u32, 0x1c456002u32, 0x6f196330u32,
    0xd94ebeafu32, 0x9cc4a611u32, 0x261dc1f2u32, 0x5815a7bdu32, 0x70b7ed67u32, 0xa1513c68u32, 0x44f93634u32, 0x720dcdfcu32
];

// http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
pub fn lds(n: u32, dimension: usize, offset: u32) -> f32 {
    const INV_U32_MAX_FLOAT: f32 = 1.0 / 4294967296.0;
    (LDS_PRIMES[dimension].wrapping_mul(n.wrapping_add(offset))) as f32 * INV_U32_MAX_FLOAT 
}

pub struct RngState {
    state: UVec2,
    dimension: usize,
}

impl RngState {
    pub fn new(state: UVec2) -> Self {
        Self {
            state,
            dimension: 0,
        }
    }

    pub fn next_state(&self) -> UVec2 {
        UVec2::new(self.state.x + 1, self.state.y)
    }

    pub fn gen_r1(&mut self) -> f32 {
        self.dimension += 1;
        lds(self.state.x, self.dimension, self.state.y)
    }

    pub fn gen_r2(&mut self) -> Vec2 {
        Vec2::new(self.gen_r1(), self.gen_r1())
    }

    pub fn gen_r3(&mut self) -> Vec3 {
        Vec3::new(self.gen_r1(), self.gen_r1(), self.gen_r1())
    }
}