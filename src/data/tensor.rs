use ndarray::prelude::*;
use ndarray::Array;
use std::cmp::Ordering;
use std::vec::Vec;

use ndarray::IxDynImpl;

use num_traits::Zero;

pub struct Tensor<T>
where
    T: Clone + Zero,
{
    raw_shapes: Vec<usize>,
    data: Array<T, Dim<IxDynImpl>>,
}

impl<T> Tensor<T>
where
    T: Clone + Zero,
{
    pub fn new(raw_shapes: &[usize]) -> Self {
        let raw_shapes = raw_shapes.to_vec();

        let data = Array::<T, _>::zeros(raw_shapes.clone());

        Tensor { raw_shapes, data }
    }

    // pub fn new(size: u32) -> Tensor<T> {
    //     Tensor {
    //         raw_shapes: Vec::new(),
    //         data: Vec::new(),
    //     }
    // }
    // pub fn new_with_dimensions(dimensions: Vec<u32>) -> Tensor {
    //     let size: usize = dimensions.iter().map(|&x| x as usize).product();
    //     Tensor {
    //         raw_shapes: dimensions.clone(),
    //         data: vec![0.0; size],
    //     }
    // }
    pub fn ndim(&self) -> usize {
        self.raw_shapes.len()
    }

    pub fn rows(&self) -> usize {
        let mut n = self.ndim();
        if n == 0 {
            n += 2;
        }
        else if n == 1 {
            n += 1;
            
        }
        self.raw_shapes.get(n - 2).copied().unwrap()
    }

    pub fn cols(&self) -> usize {
        let mut n = self.ndim();
        if n == 0 {
            n += 2;
        }
        self.raw_shapes.get(n - 1).copied().unwrap()
    }

    pub fn channels(&self) -> &Vec<usize> {
        &self.raw_shapes
    }

    pub fn size(&self) -> usize {
        self.raw_shapes.iter().product()
    }

    // pub fn set_data(&mut self, new_data: Vec<f32>) {
    //     self.data = new_data;
    // }

    // pub fn empty(&self) -> bool {
    //     self.data.is_empty()
    // }

    // pub fn index(&self, offset: usize) -> f32 {
    //     self.data.get(offset).copied().unwrap_or(0.0)
    // }

    // pub fn index_mut(&mut self, offset: usize) -> &mut f32 {
    //     self.data.get_mut(offset).unwrap_or(&mut 0.0)
    // }

    pub fn raw_shapes(&self) -> &Vec<usize> {
        &self.raw_shapes
    }

    // pub fn data(&self) -> &[f32] {
    //     &self.data
    // }

    // pub fn data_mut(&mut self) -> &mut [f32] {
    //     &mut self.data
    // }

    // pub fn slice(&self, channel: usize) -> &[f32] {
    //     let start = channel * self.rows() as usize * self.cols() as usize;
    //     &self.data[start..(start + self.rows() as usize * self.cols() as usize)]
    // }

    // More methods can be translated similarly...
}
