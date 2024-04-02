use ndarray::LinalgScalar;
use num_traits::{Bounded, Zero};

use super::Tensor;
use crate::data::SharedTensor;
use std::cell::RefCell;
use std::rc::Rc;

pub fn create_tensor<A: Clone + Zero>(
    channels: usize,
    rows: usize,
    cols: usize,
) -> SharedTensor<A> {
    Rc::new(RefCell::new(Tensor::new(&[channels, rows, cols])))
}
pub fn create_tensor_2<A: Clone + Zero>(rows: usize, cols: usize) -> SharedTensor<A> {
    Rc::new(RefCell::new(Tensor::new(&[rows, cols])))
}
pub fn create_tensor_1<A: Clone + Zero>(cols: usize) -> SharedTensor<A> {
    Rc::new(RefCell::new(Tensor::new(&[cols])))
}
pub fn tensor_element_add<A: Clone + Zero>(
    tensor1: &SharedTensor<A>,
    tensor2: &SharedTensor<A>,
) -> SharedTensor<A> {
    let out_data = tensor1.as_ref().borrow().data() + tensor2.as_ref().borrow().data();
    Tensor::from_ndarry(out_data).shared_self()
}
pub fn tensor_element_multiply<A: Clone + Zero>(
    tensor1: &SharedTensor<A>,
    tensor2: &SharedTensor<A>,
) -> SharedTensor<A>
where
    A: Clone + LinalgScalar + PartialOrd + Bounded + std::ops::Neg + std::fmt::Debug,
{
    let out_data = tensor1.as_ref().borrow().data() * tensor2.as_ref().borrow().data();
    Tensor::from_ndarry(out_data).shared_self()
}
#[cfg(test)]
mod test_torch_utils {
    use ndarray::Array2;

    use super::*;
    #[test]
    fn test_create_tensor() {
        let tensor = create_tensor::<f32>(4, 128, 128);
        // assert_eq!(tensor.as_ref().borrow().channels(), [4, 128, 128]);
        println!("{:?}", tensor);
    }
    #[test]
    fn test() {
        // 创建两个相同的矩阵
        let a = Array2::<f64>::ones((2, 2));
        let b = Array2::<f64>::ones((2, 2));

        // 对应元素相乘
        let result = &a * &b;

        println!("{:?}", result);
    }
}
