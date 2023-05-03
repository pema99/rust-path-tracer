use core::ops::Index;

pub struct FixedVec<T, const CAPACITY: usize> {
    pub data: [T; CAPACITY],
    pub len: u32,
}

impl<T: Sized + Default + Copy, const CAPACITY: usize> FixedVec<T, CAPACITY> {
    pub fn new() -> Self {
        Self {
            data: [Default::default(); CAPACITY],
            len: 0,
        }
    }

    pub fn push(&mut self, value: T) {
        self.data[self.len as usize] = value;
        self.len += 1;
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len > 0 {
            self.len -= 1;
            Some(self.data[self.len as usize])
        } else {
            None
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.len = 0;
    }
}

impl<T, const CAPACITY: usize> Index<u32> for FixedVec<T, CAPACITY> {
    type Output = T;

    fn index(&self, index: u32) -> &Self::Output {
        &self.data[index as usize]
    }
}