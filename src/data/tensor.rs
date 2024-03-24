use ndarray::prelude::*;
use ndarray::Array;
use ndarray::Shape;
use num_traits::Zero;
use ndarray::ArrayD;
use ndarray::IxDyn;
use std::cell::RefCell;
use std::rc::Rc;
pub type SharedTensor<A> = Rc<RefCell<Tensor<A>>>;
#[derive(Debug)]
pub struct Tensor<A> {
    raw_shapes: IxDyn,
    data: ArrayD<A>,
}

impl<A> Tensor<A>
where
    A: Clone + Zero,
{
    pub fn new(shape: &[usize]) -> Tensor<A> {
        let raw_shapes = IxDyn(shape);
        let data: ArrayD<A> = ArrayD::<A>::zeros(raw_shapes.clone());

        Tensor { raw_shapes, data }
    }
    // }
    // pub fn new_with_dimensions(dimensions: Vec<u32>) -> Tensor {
    //     let size: usize = dimensions.iter().map(|&x| x as usize).product();
    //     Tensor {
    //         raw_shapes: dimensions.clone(),
    //         data: vec![0.0; size],
    //     }
    // }
    pub fn ndim(&self) -> usize {
        self.raw_shapes.ndim()
    }

    pub fn rows(&self) -> usize {
        let ndim = self.ndim();
        if ndim > 1 {
            self.raw_shapes[ndim - 2]
        } else if ndim == 1 {
            1
        } else {
            // Handle the case when n is 0 or negative
            panic!("Invalid state: ndim() returned 0, which is not allowed.");
        }
    }

    pub fn cols(&self) -> usize {
        let ndim = self.ndim();
        if ndim == 0 {
            // Handle the case when n is 0
            panic!("Invalid state: ndim() returned 0, which is not allowed for cols().");
        }
        else {
            self.raw_shapes[ndim - 1]
        }
        
    }

    pub fn channels(&self) -> &IxDyn {
        &self.raw_shapes
    }

    pub fn size(&self) -> usize {
        self.raw_shapes.size()
    }

    pub fn set_data(&mut self, new_data: ArrayD<A>) {
        self.data = new_data;
    }

    // pub fn empty(&self) -> bool {
    //     self.data.is_empty()
    // }

    // pub fn index(&self, offset: usize) -> f32 {
    //     self.data.get(offset).copied().unwrap_or(0.0)
    // }

    // pub fn index_mut(&mut self, offset: usize) -> &mut f32 {
    //     self.data.get_mut(offset).unwrap_or(&mut 0.0)
    // }

    pub fn shapes(&self) -> Vec<usize> {
        let mut shape:Vec<usize> = Vec::new();
        for i in 0..self.raw_shapes.ndim(){
            shape.push(self.raw_shapes[i]);
        }
        shape
    }

    pub fn data(&self) -> &ArrayD<A> {
        &self.data
    }

    // pub fn data_mut(&mut self) -> &mut [f32] {
    //     &mut self.data
    // }


    pub fn slice<I>(&self, info:I) -> ArrayView<'_, A, I::OutDim>
    where
        I: ndarray::SliceArg<IxDyn>  + std::fmt::Debug
    {
        self.data.view().slice_move(info)
    }
    
}

// extern crate kuiper_infer;



#[cfg(test)]
mod test_tensor {
    use super::*;
    use ndarray::prelude::*;
    #[test]
    fn test_new_tensor() {
        let _tensor = Tensor::<f32>::new( & [1, 2, 5]);

    }
    #[test]
    fn test_tensor_init1(){
        let f1 = Tensor::<f32>::new(&[3, 224, 224]);

        let mut x = [3, 224, 224].f();
        x.set_f(false);
        print!("{:?}", x);

        assert_eq!(f1.channels().ndim(),3);
        assert_eq!(f1.rows(), 224);
        assert_eq!(f1.cols(), 224);
        assert_eq!(f1.size(), 224 * 224 * 3);

    }
    #[test]
    fn test_tensor_init2(){

        let f1 = Tensor::<u32>::new(& [3, 224, 224]);

        assert_eq!(f1.channels().ndim(), 3);
        assert_eq!(f1.rows(), 224);
        assert_eq!(f1.cols(), 224);
        assert_eq!(f1.size(), 224 * 224 * 3);

    }
    #[test]
    fn test_tensor_init3(){
        let f1 = Tensor::<u32>::new(&[1, 13, 14]);


        assert_eq!(f1.channels().ndim(), 3);
        assert_eq!(f1.rows(), 13);
        assert_eq!(f1.cols(), 14);
        assert_eq!(f1.size(), 13 * 14);

    }
    #[test]
    fn test_tensor_init4(){
        let f1 = Tensor::<u32>::new(&[13, 15]);


        assert_eq!(f1.ndim(), 2);
        assert_eq!(f1.rows(), 13);
        assert_eq!(f1.cols(), 15);
        assert_eq!(f1.size(), 13 * 15);

    }
    #[test] 
    fn test_tensor_init5(){

        let f1 = Tensor::<u32>::new(&[16, 13, 15]);


        assert_eq!(f1.ndim(), 3);
        assert_eq!(f1.rows(), 13);
        assert_eq!(f1.cols(), 15);
        assert_eq!(f1.size(), 13 * 16 * 15);

    }

    #[test]
    fn test_tensor_init_1d() {
        let f1 = Tensor::<f32>::new(&[3]);

        assert_eq!(f1.ndim(), 1);
        assert_eq!(f1.channels()[0], 3);
    }

    #[test]
    fn test_tensor_init_2d() {

        let f1 = Tensor::<f32>::new(&[32, 24]);
        let raw_shapes = f1.channels();

        assert_eq!(raw_shapes.ndim(), 2);
        assert_eq!(raw_shapes[0], 32);
        assert_eq!(raw_shapes[1], 24);
    }
    #[test]
    fn test_tensor_data(){

        let f1 = Tensor::<f32>::new(&[32, 24]);
        let data = f1.data();
        println!("{:?}", data);
    }

    #[test]
    fn test_set_data(){
  
        let mut f1 = Tensor::<f32>::new(& [3, 224, 224]);
        let data = ArrayD::<f32>::zeros(IxDyn(&[3, 224, 224]));

        f1.set_data(data);

    }
    #[test]
    fn test_slice(){
        let mut f1 = Tensor::<f32>::new(&[3, 224, 224]);

        println!("Data in the first channel: {:?}", f1.slice(s![0, .., ..]).raw_dim()); 
        println!("Data in the (1,1,1): {}", f1.data()[[1, 1, 1]]);
    }




}
