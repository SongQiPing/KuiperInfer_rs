use std::rc::Rc;

use crate::data::SharedTensor;
use crate::layer::abstract_layer::layer::*;
use crate::layer::abstract_layer::RuntimeOperatorData;
use crate::layer::Layer;
use log::error;
use num_traits::Bounded;
use num_traits::Zero;

use crate::layer::LayerError;
use crate::runtime::SharedRuntimeOperator;
pub struct MaxPoolingLayer<A> {
    runtime_operator: RuntimeOperatorData<A>,
    layer_name: String,
    padding_h: usize,
    padding_w: usize,
    pooling_size_h: usize,
    pooling_size_w: usize,
    stride_h: usize,
    stride_w: usize,
}

impl<A> MaxPoolingLayer<A>
where
    A: Clone + Zero + PartialOrd + std::ops::Neg + 'static + Bounded,
    f32: From<A>,
    A: From<f32>,
{
    fn new(runtime_operator: RuntimeOperatorData<A>) -> Self {
        MaxPoolingLayer {
            runtime_operator: runtime_operator,
            layer_name: "nn.MaxPool2d".to_string(),
            padding_h: 0,
            padding_w: 0,
            pooling_size_h: 0,
            pooling_size_w: 0,
            stride_h: 1,
            stride_w: 1,
        }
    }
    pub fn get_instance(runtime_operator: SharedRuntimeOperator<A>) -> Rc<dyn Layer<A>> {
        // let params_map = &runtime_operator.as_ref().borrow().params;
        // if let None = params_map.get(&"stride".to_string()) {
        //     panic!();
        // }

        // match params_map.get("stride".to_string()) {
        //     Some() => {}
        // }
        Rc::new(Self::new(RuntimeOperatorData::new_from(runtime_operator)))
    }

    fn batch_forward_wight_tensors(
        &self,
        inputs: &SharedTensor<A>,
        outputs: &SharedTensor<A>,
    ) -> Result<(), LayerError> {
        let input_h = inputs.as_ref().borrow().rows();
        let input_w = inputs.as_ref().borrow().cols();
        let input_c = inputs.as_ref().borrow().channels();
        let (output_h, output_w) = self.calc_output_shape(input_h, input_w);
        // 检验输出的形状是否符合要求
        if outputs.borrow().rows() != output_h || outputs.borrow().cols() != output_w {
            return Err(LayerError::InferFailedOutputSizeError);
        }
        let input_padded_h = input_h + 2 * self.padding_h;
        let input_padded_w = input_w + 2 * self.padding_w;
        for channel_index in 0..input_c {
            for c in (0..=input_padded_w - self.pooling_size_w).step_by(self.stride_w) {
                for r in (0..=input_padded_h - self.pooling_size_h).step_by(self.stride_h) {
                    let mut max_val = A::min_value();
                    for w in 0..self.pooling_size_w {
                        for h in 0..self.pooling_size_h {
                            let mut currect_val: A = Zero::zero();
                            if h + r >= self.padding_h
                                && w + c >= self.padding_w
                                && h + r < input_c + self.padding_w
                                && w + c < input_w + self.padding_w
                            {
                                currect_val = inputs
                                    .as_ref()
                                    .borrow()
                                    .index(&[
                                        channel_index,
                                        r + h - self.padding_h,
                                        w + c - self.padding_w,
                                    ])
                                    .clone();
                            }

                            max_val = partial_min_max::max(max_val, currect_val);
                        }
                    }
                    let output_row = r / self.stride_h;
                    let output_col = c / self.stride_w;
                    *outputs
                        .borrow_mut()
                        .index_mut(&[channel_index, output_row, output_col]) = max_val;
                }
            }
        }
        Ok(())
    }
    /// 计算运算之后的形状
    fn calc_output_shape(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let padding_h = self.padding_h;
        let pooling_h_size = self.pooling_size_h;
        let stride_h = self.stride_h;
        let output_h: usize = (input_h - pooling_h_size + 2 * padding_h) / stride_h + 1;

        let padding_w = self.padding_w;
        let pooling_w_size = self.pooling_size_w;
        let stride_w = self.stride_w;
        let output_w: usize = (input_w - pooling_w_size + 2 * padding_w) / stride_w + 1;
        (output_h, output_w)
    }
}
use std::convert::From;

impl<A> Layer<A> for MaxPoolingLayer<A>
where
    A: Clone + Zero + std::ops::Neg + 'static + PartialOrd + Bounded,
    A: From<f32>,
    f32: From<A>,
{
    fn forward(&self) -> Result<(), LayerError> {
        let layer_input_datas = self.runtime_operator.prepare_input_tensor();
        let layer_ouput_datas = self.runtime_operator.prepare_output_tensor();

        if let Err(e) = self.check_inputs_and_outputs(&layer_input_datas, &layer_ouput_datas) {
            return Err(e);
        }
        if self.stride_h == 0 || self.stride_w == 0 {
            error!("The stride parameter is set incorrectly. It must always be greater than 0");
            return Err(LayerError::InferFailedStrideParameterError);
        }

        if let Err(e) = self.forward_with_tensors(&layer_input_datas, &layer_ouput_datas) {
            return Err(e);
        }

        Ok(())
    }
    fn forward_with_tensors(
        &self,
        inputs: &Vec<SharedTensor<A>>,
        outputs: &Vec<SharedTensor<A>>,
    ) -> Result<(), LayerError> {
        let batch_size = inputs.len();
        for i in 0..batch_size {
            let input_data = &inputs[i];
            let output_data = &outputs[i];
            if let Err(e) = self.batch_forward_wight_tensors(input_data, output_data) {
                return Err(e);
            }
        }
        Ok(())
    }

    fn layer_name(&self) -> &String {
        &self.layer_name
    }
}

#[cfg(test)]
mod test_sigmoid_layer {

    use crate::data::Tensor;
    use crate::layer::LayerRegisterer;
    use ndarray::ArrayD;
    use ndarray::IxDyn;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_create_layer_find() {
        // 检查nn.ReLu 算子是否注册
        let layer_type = "nn.Sigmoid".to_string();
        assert!(LayerRegisterer::check_operator_registration(&layer_type));
    }
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    fn are_equal(a: f32, b: f32) -> bool {
        // 定义一个阈值来接受误差范围
        let epsilon = 1e-3;
        // 比较两个浮点数的绝对差值是否小于阈值
        (a - b).abs() < epsilon
    }
    #[test_log::test]
    fn test_create_layer_reluforward() {
        use crate::layer::LayerRegisterer;
        use crate::runtime::RuntimeOperator;
        use log::info;
        use std::cell::RefCell;
        use std::rc::Rc;
        let mut runtime_operator = RuntimeOperator::<f32>::new();
        runtime_operator.type_name = "nn.Sigmoid".to_string();
        let runtime_operator = Rc::new(RefCell::new(runtime_operator));
        let sigmoid_layer = LayerRegisterer::create_layer(&runtime_operator);

        let mut input_data = Tensor::<f32>::new(&[3, 4, 4]);
        let out_data = Tensor::<f32>::new(&[3, 4, 4]);
        input_data.data = ArrayD::random(IxDyn(&[3, 4, 4]), Uniform::new(-5., 5.0));

        info!("{:?}", input_data);
        let input_data = vec![Rc::new(RefCell::new(input_data))];
        let out_data = vec![Rc::new(RefCell::new(out_data))];
        sigmoid_layer
            .forward_with_tensors(&input_data, &out_data)
            .unwrap();
        info!("-------input_data-------");
        info!("{:?}", input_data);
        info!("-------output_data");
        info!("{:?}", out_data);

        for (element1, element2) in input_data[0]
            .borrow()
            .data
            .iter()
            .zip(out_data[0].borrow().data.iter())
        {
            println!("Array1 element: {}, Array2 element: {}", element1, element2);
            assert!(are_equal(sigmoid(*element1), *element2));
        }
    }
}
