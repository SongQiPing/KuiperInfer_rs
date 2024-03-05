extern crate kuiper_infer;

#[cfg(test)]
mod test_tensor {

    #[test]
    fn test_new_tensor() {
        use ndarray::prelude::*;
        let _tensor = kuiper_infer::Tensor::<f32, _>::new( [1, 2, 5].f());

    }
    #[test]
    fn test_tensor_init1(){
        use ndarray::prelude::*;
        use  kuiper_infer::Tensor;
        let f1 = Tensor::<f32, _>::new([3, 224, 224].f());

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
        use ndarray::prelude::*;
        use  kuiper_infer::Tensor;
        let f1 = Tensor::<u32, _>::new([3, 224, 224].f());


        assert_eq!(f1.channels().ndim(), 3);
        assert_eq!(f1.rows(), 224);
        assert_eq!(f1.cols(), 224);
        assert_eq!(f1.size(), 224 * 224 * 3);

    }
    #[test]
    fn test_tensor_init3(){
        use ndarray::prelude::*;
        use  kuiper_infer::Tensor;
        let f1 = Tensor::<u32, _>::new([1, 13, 14].f());


        assert_eq!(f1.channels().ndim(), 3);
        assert_eq!(f1.rows(), 13);
        assert_eq!(f1.cols(), 14);
        assert_eq!(f1.size(), 13 * 14);

    }
    #[test]
    fn test_tensor_init4(){
        use ndarray::prelude::*;
        use  kuiper_infer::Tensor;
        let f1 = Tensor::<u32, _>::new([13, 15].f());


        assert_eq!(f1.ndim(), 2);
        assert_eq!(f1.rows(), 13);
        assert_eq!(f1.cols(), 15);
        assert_eq!(f1.size(), 13 * 15);

    }
    #[test] 
    fn test_tensor_init5(){
        use ndarray::prelude::*;
        use  kuiper_infer::Tensor;
        let f1 = Tensor::<u32, _>::new([16, 13, 15].f());


        assert_eq!(f1.ndim(), 3);
        assert_eq!(f1.rows(), 13);
        assert_eq!(f1.cols(), 15);
        assert_eq!(f1.size(), 13 * 16 * 15);

    }

    #[test]
    fn test_tensor_init_1d() {
        use ndarray::prelude::*;
        use kuiper_infer::Tensor;
        let f1 = Tensor::<f32, _>::new([3].f());

        assert_eq!(f1.ndim(), 1);
        assert_eq!(f1.channels()[0], 3);
    }

    #[test]
    fn test_tensor_init_2d() {

        use ndarray::prelude::*;

        let f1 = kuiper_infer::Tensor::<f32, _>::new([32, 24].f());
        let raw_shapes = f1.channels();

        assert_eq!(raw_shapes.ndim(), 2);
        assert_eq!(raw_shapes[0], 32);
        assert_eq!(raw_shapes[1], 24);
    }
    #[test]
    fn test_tensor_data(){
        use ndarray::prelude::*;

        let f1 = kuiper_infer::Tensor::<f32, _>::new([32, 24].f());
        let data = f1.data();
        println!("{:?}", data);
    }

    #[test]
    fn test_set_data(){
        use ndarray::prelude::*;
        use ndarray::Array;
  
        let mut f1 = kuiper_infer::Tensor::<f32, _>::new([3, 224, 224].f());
        let data = Array::<f32, _>::zeros((3, 224, 224).f());

        f1.set_data(data);

    }

    // #[test]
    // fn test_ndarry(){
    //     use ndarray::prelude::*;
    //     use ndarray::Array;

    //     let a = Array::range(0., 10., 1.);

    //     let mut a = a.mapv(|a: f64| a.powi(3));  // numpy equivlant of `a ** 3`; https://doc.rust-lang.org/nightly/std/primitive.f64.html#method.powi
    
    //     println!("{}", a);
    
    //     println!("{}", a[[2]]);
    //     println!("{}", a.slice(s![2]));
    
    //     println!("{}", a.slice(s![2..5]));
    
    //     a.slice_mut(s![..6;2]).fill(1000.);  // numpy equivlant of `a[:6:2] = 1000`
    //     println!("{}", a);
    
    //     for i in a.iter() {
    //         print!("{}, ", i.powf(1./3.))
    //     }

    // }


}
