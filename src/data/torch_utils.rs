use num_traits::Zero;

use super::Tensor;
use std::rc::Rc;
use std::cell::RefCell;
use crate::data::SharedTensor;

pub fn create_tensor<A:Clone+Zero>(channels:usize, rows:usize, cols:usize) -> SharedTensor<A>{

    Rc::new(RefCell::new(Tensor::new(&[channels, rows, cols])))
}
pub fn create_tensor_2<A:Clone+Zero>(rows:usize, cols:usize) -> SharedTensor<A>{

    Rc::new(RefCell::new(Tensor::new(&[rows, cols])))
}
pub fn create_tensor_1<A:Clone+Zero>(cols:usize) -> SharedTensor<A>{

    Rc::new(RefCell::new(Tensor::new(&[cols])))
}

#[cfg(test)]
mod test_torch_utils{
    use super::*;
    #[test]
    fn test_create_tensor(){
        let tensor = create_tensor::<f32>(4, 128, 128);
        // assert_eq!(tensor.as_ref().borrow().channels(), [4, 128, 128]);
        println!("{:?}", tensor);
        
    }


}