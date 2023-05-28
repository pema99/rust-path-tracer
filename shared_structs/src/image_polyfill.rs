// This file contains a polyfill for the Image and Sampler types, which are not available on the CPU.

#[cfg(target_arch = "spirv")]
pub mod polyfill {
    pub use spirv_std::{Sampler, Image};
}

#[cfg(not(target_arch = "spirv"))]
pub mod polyfill {
    use glam::{Vec4, Vec2, IVec2};

    #[derive(Clone, Copy)]
    pub struct Sampler;

    pub struct Image<'a, A,B,C,D,E,F> {
        _phantom: core::marker::PhantomData<(A,B,C,D,E,F)>,
        width: u32,
        height: u32,
        buffer: &'a [Vec4],
    }

    impl<'a, A> Image<'a, A,A,A,A,A,A> {
        pub const fn new (buffer: &'a [Vec4], width: u32, height: u32) -> Self {
            Image {
                _phantom: core::marker::PhantomData,
                width,
                height,
                buffer,
            }
        }

        fn sample_raw(&self, coord: IVec2) -> Vec4 {
            let x = coord.x as usize % self.width as usize;
            let y = coord.y as usize % self.height as usize;
            self.buffer[y * self.width as usize + x]
        }

        pub fn sample_by_lod(&self, _sampler: Sampler, coord: Vec2, _lod: f32) -> Vec4 {
            let scaled_uv = coord * Vec2::new(self.width as f32, self.height as f32);
            let frac_uv = scaled_uv.fract();
            let ceil_uv = scaled_uv.ceil().as_ivec2();
            let floor_uv = scaled_uv.floor().as_ivec2();

            // Bilinear filtering
            let c00 = self.sample_raw(floor_uv);
            let c01 = self.sample_raw(IVec2::new(floor_uv.x, ceil_uv.y));
            let c10 = self.sample_raw(IVec2::new(ceil_uv.x, floor_uv.y));
            let c11 = self.sample_raw(ceil_uv);
            let tx = frac_uv.x;
            let ty = frac_uv.y;

            let a = c00.lerp(c10, tx);
            let b = c01.lerp(c11, tx);
            a.lerp(b, ty)
        }
    }

    #[macro_export]
    macro_rules! Image {
        ($a:expr, $b:ident=$d:ident, $c:expr) => { Image<(), (), (), (), (), ()> };
    }

    pub type CpuImage<'fw> = Image<'fw, (),(),(),(),(),()>;
}