pub mod status_code;

pub mod data;
use std::ops::{Add, Div, Mul, Sub};

pub use data::tensor::Tensor;

pub mod runtime;
use ndarray::LinalgScalar;
use num_traits::{One, Zero};
pub use runtime::pnnx;

pub mod layer;

pub mod parser;

// pub trait TensorScalar: LinalgScalar {}
pub trait TensorScalar:
    'static
    + Copy
    + Zero
    + One
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
}

impl<T> TensorScalar for T where
    T: 'static
        + Copy
        + Zero
        + One
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
{
}
