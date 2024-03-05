use ndarray::prelude::*;
use ndarray::Array;
use ndarray::Shape;
use ndarray::RawData;
use num_traits::Zero;

pub struct Tensor<A, D: Dimension> {
    raw_shapes: D,
    data: Array<A, D>,
}

impl<A, D> Tensor<A, D>
where
    A: Clone + Zero,
    // S: RawData<Elem = A>,
    D: Dimension,
{
    pub fn new(shape: Shape<D>) -> Tensor<A, D> {
        let raw_shapes = shape.raw_dim().clone();
        let data: ArrayBase<ndarray::OwnedRepr<A>, D> = Array::<A, D>::zeros(shape);

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

    pub fn channels(&self) -> &D {
        &self.raw_shapes
    }

    pub fn size(&self) -> usize {
        self.raw_shapes.size()
    }

    pub fn set_data(&mut self, new_data: Array<A, D>) {
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

    // pub fn raw_shapes(&self) -> &Vec<usize> {
    //     &self.raw_shapes
    // }

    pub fn data(&self) -> &Array<A, D> {
        &self.data
    }

    // pub fn data_mut(&mut self) -> &mut [f32] {
    //     &mut self.data
    // }


    // pub fn slice<I>(&self, info: I) -> ArrayView<'_, A, I::OutDim>
    // where
    //     I: ndarray::SliceArg<D>,
    //     S: ndarray:: Data,
    // {
    //     self.data.view().slice_move(info)
    // }
    // More methods can be translated similarly...
}
