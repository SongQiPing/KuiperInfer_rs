extern crate kuiper_infer;

#[cfg(test)]
mod test_tensor {

    #[test]
    fn test_new_tensor() {
        let _tensor = kuiper_infer::Tensor::<f32>::new(&[1, 2, 5]);
    }
    #[test]
    fn test_tensor_init1(){
        use  kuiper_infer::Tensor;
        let f1 = Tensor::<f32>::new(&[3, 224, 224]);

        assert_eq!(f1.channels().len(), 3);
        assert_eq!(f1.rows(), 224);
        assert_eq!(f1.cols(), 224);
        assert_eq!(f1.size(), 224 * 224 * 3);

    }
    #[test]
    fn test_tensor_init2(){
        use  kuiper_infer::Tensor;
        let f1 = Tensor::<u32>::new(&[3, 224, 224]);


        assert_eq!(f1.channels().len(), 3);
        assert_eq!(f1.rows(), 224);
        assert_eq!(f1.cols(), 224);
        assert_eq!(f1.size(), 224 * 224 * 3);

    }
    #[test]
    fn test_tensor_init3(){
        use  kuiper_infer::Tensor;
        let f1 = Tensor::<u32>::new(&[1, 13, 14]);


        assert_eq!(f1.channels().len(), 3);
        assert_eq!(f1.rows(), 13);
        assert_eq!(f1.cols(), 14);
        assert_eq!(f1.size(), 13 * 14);

    }
    #[test]
    fn test_tensor_init4(){
        use  kuiper_infer::Tensor;
        let f1 = Tensor::<u32>::new(&[13, 15]);


        assert_eq!(f1.channels().len(), 3);
        assert_eq!(f1.rows(), 13);
        assert_eq!(f1.cols(), 14);
        assert_eq!(f1.size(), 13 * 14);

    }
    #[test] 
    fn test_tensor_init5(){
        use  kuiper_infer::Tensor;
        let f1 = Tensor::<u32>::new(&[16, 13, 15]);


        assert_eq!(f1.channels().len(), 3);
        assert_eq!(f1.rows(), 13);
        assert_eq!(f1.cols(), 15);
        assert_eq!(f1.size(), 13 * 16 * 15);

    }
    fn test_copy_construct1(){
        use  kuiper_infer::Tensor;
        let f1 = Tensor::<u32>::new(&[3, 224, 224]);

    }
    #[test]
    fn test_tensor_init_1d() {
        use kuiper_infer::Tensor;
        let f1 = Tensor::<f32>::new(&[3]);
        let raw_shapes = f1.raw_shapes();
        assert_eq!(f1.channels().len(), 1);
        assert_eq!(raw_shapes[0], 3);
    }

    #[test]
    fn test_tensor_init_2d() {

        let f1 = kuiper_infer::Tensor::<f32>::new(&[32, 24]);
        let raw_shapes = f1.raw_shapes();

        assert_eq!(raw_shapes.len(), 2);
        assert_eq!(raw_shapes[0], 32);
        assert_eq!(raw_shapes[1], 24);
    }
    #[test]
    fn test_tensor_data(){
        let f1 = kuiper_infer::Tensor::<f32>::new(&[32, 24]);
        let data = f1.data();
        println!("{:?}", data);
    }

    #[test]
    fn test_set_data(){
        use ndarray::prelude::*;
        use ndarray::Array;
        use kuiper_infer::Tensor;
        let f1 = Tensor::<f32>::new(&[3, 224, 224]);
        let data = Array::<f32, _>::zeros((3, 224, 224).f());

        f1.set_data(data);
        

    }

    #[test]
    fn test_ndarry(){
        use ndarray::prelude::*;
        use ndarray::Array;

        let a = Array::range(0., 10., 1.);

        let mut a = a.mapv(|a: f64| a.powi(3));  // numpy equivlant of `a ** 3`; https://doc.rust-lang.org/nightly/std/primitive.f64.html#method.powi
    
        println!("{}", a);
    
        println!("{}", a[[2]]);
        println!("{}", a.slice(s![2]));
    
        println!("{}", a.slice(s![2..5]));
    
        a.slice_mut(s![..6;2]).fill(1000.);  // numpy equivlant of `a[:6:2] = 1000`
        println!("{}", a);
    
        for i in a.iter() {
            print!("{}, ", i.powf(1./3.))
        }

    }


}
