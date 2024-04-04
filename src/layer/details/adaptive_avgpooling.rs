use std::rc::Rc;

use crate::data::SharedTensor;
use crate::layer::abstract_layer::layer::*;
use crate::layer::abstract_layer::RuntimeOperatorData;
use crate::layer::Layer;
use crate::layer::ParameterData;
use crate::TensorScalar;

use log::error;
use log::info;
use num_traits::Bounded;
use num_traits::Zero;

use crate::layer::LayerError;
use crate::runtime::SharedRuntimeOperator;

pub struct AdaptiveAveragePoolingLayer<A> {
    runtime_operator: RuntimeOperatorData<A>,
    layer_name: String,
    output_h: usize,
    output_w: usize,
}

impl<A> AdaptiveAveragePoolingLayer<A>
where
    A: TensorScalar + std::fmt::Debug,
    f32: From<A>,
    A: From<f32>,
{
    fn new(output_h: usize, output_w: usize) -> Self {
        AdaptiveAveragePoolingLayer {
            runtime_operator: RuntimeOperatorData::new(),
            layer_name: "nn.MaxPool2d".to_string(),
            output_h,
            output_w,
        }
    }

    fn calculate_maxpool_window_value(
        &self,
        inputs: &SharedTensor<A>,
        cur_channel: usize,
        row_start: usize,
        col_start: usize,
        pooling_h: usize,
        pooling_w: usize,
    ) -> A {
        let input_h = inputs.as_ref().borrow().rows();
        let input_w = inputs.as_ref().borrow().cols();
        let mut sum_val = A::zero();
        for r in 0..pooling_h {
            for c in 0..pooling_w {
                let mut currect_val: A = Zero::zero();
                let cur_row = row_start + r;
                let cur_col = col_start + c;

                currect_val = inputs
                    .as_ref()
                    .borrow()
                    .index(&[cur_channel, cur_row, cur_col])
                    .clone();

                sum_val = sum_val + currect_val;
            }
        }
        sum_val
    }
    pub fn get_instance(
        runtime_operator: SharedRuntimeOperator<A>,
    ) -> Result<Rc<dyn Layer<A>>, LayerError> {
        let params_map = &runtime_operator.as_ref().borrow().params;
        //获取stride的参数
        let output_hw = ParameterData::output_hw(params_map)?;

        let adaptive_average_pool_layer = AdaptiveAveragePoolingLayer {
            runtime_operator: RuntimeOperatorData::new_from(runtime_operator.clone()),
            layer_name: "nn.AdaptiveAvgPool2d".to_string(),
            output_h: output_hw[0].try_into().unwrap(),
            output_w: output_hw[1].try_into().unwrap(),
        };
        info!("AdaptiveAveragePoolingLayer 参数:");
        info!("Layer Name: {}", adaptive_average_pool_layer.layer_name);
        info!(
            "OutPut shapt (H, W): ({}, {})",
            adaptive_average_pool_layer.output_h, adaptive_average_pool_layer.output_w
        );

        Ok(Rc::new(adaptive_average_pool_layer))
    }

    fn batch_forward_wight_tensors(
        &self,
        inputs: &SharedTensor<A>,
        outputs: &SharedTensor<A>,
    ) -> Result<(), LayerError> {
        let input_h = inputs.as_ref().borrow().rows();
        let input_w = inputs.as_ref().borrow().cols();
        let input_c = inputs.as_ref().borrow().channels();
        let output_h = self.output_h;
        let output_w = self.output_w;

        let stride_h = input_h / self.output_h;
        let stride_w = input_w / self.output_w;
        assert!(
            stride_h > 0 && stride_w > 0,
            "The stride parameter is set incorrectly. It must always be greater than 0"
        );

        let pooling_h = input_h - (output_h - 1) * stride_h;
        let pooling_w = input_w - (output_w - 1) * stride_w;

        // 检验输出的形状是否符合要求
        if outputs.borrow().rows() != self.output_h || outputs.borrow().cols() != self.output_w {
            error!(
                "now output data shape is {:?}, expect outdata shape is{:?} ",
                (outputs.borrow().rows(), outputs.borrow().cols()),
                (self.output_h, self.output_w)
            );

            return Err(LayerError::InferFailedOutputSizeError);
        }
        let pooling_size = (pooling_h * pooling_w) as f32;

        for channel_index in 0..input_c {
            for c in (0..=input_h - pooling_h).step_by(stride_h) {
                for r in (0..=input_w - pooling_w).step_by(stride_w) {
                    let sum_val = self.calculate_maxpool_window_value(
                        inputs,
                        channel_index,
                        r,
                        c,
                        pooling_h,
                        pooling_w,
                    );

                    let mean_val = sum_val / A::from(pooling_size);
                    let output_row = r / stride_h;
                    let output_col = c / stride_w;
                    *outputs
                        .borrow_mut()
                        .index_mut(&[channel_index, output_row, output_col]) = mean_val;
                }
            }
        }

        Ok(())
    }
}
use std::convert::From;

impl<A> Layer<A> for AdaptiveAveragePoolingLayer<A>
where
    A: TensorScalar + std::fmt::Debug,
    f32: From<A>,
    A: From<f32>,
{
    fn forward(&self) -> Result<(), LayerError> {
        let layer_input_datas = self.runtime_operator.prepare_input_tensor();
        let layer_ouput_datas = self.runtime_operator.prepare_output_tensor();

        if let Err(e) = self.check_inputs_and_outputs(&layer_input_datas, &layer_ouput_datas) {
            return Err(e);
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
    use crate::pnnx::Parameter;
    use crate::runtime::RuntimeOperator;
    use crate::runtime::SharedRuntimeOperator;
    use log::info;
    use ndarray::ArrayD;
    use ndarray::IxDyn;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn get_test_adaptive_maxpooling_operator() -> SharedRuntimeOperator<f32> {
        let mut maxpooling_operator = RuntimeOperator::<f32>::new();
        maxpooling_operator.type_name = "nn.AdaptiveAvgPool2d".to_string();
        maxpooling_operator.params.insert(
            "output_size".to_string(),
            Rc::new(RefCell::new(Parameter::IntList(vec![2, 2]))),
        );

        Rc::new(RefCell::new(maxpooling_operator))
    }

    #[test]
    fn test_create_layer_find() {
        // 检查nn.MaxPool2d 算子是否注册
        let layer_type = "nn.AdaptiveAvgPool2d".to_string();
        assert!(LayerRegisterer::check_operator_registration(&layer_type));
    }
    #[test]
    fn test_create_layer_poolingforward() {
        let runtime_operator = get_test_adaptive_maxpooling_operator();
        let _layer = LayerRegisterer::create_layer(&runtime_operator);
    }
    #[test_log::test]
    fn test_create_layer_poolingforward_1() {
        let runtime_operator = get_test_adaptive_maxpooling_operator();
        let layer = LayerRegisterer::create_layer(&runtime_operator);

        let mut input_data = Tensor::<f32>::new(&[1, 4, 4]);
        input_data.data = ArrayD::random(IxDyn(&[1, 4, 4]), Uniform::new(-5., 5.0));
        let mut out_data = Tensor::<f32>::new(&[1, 2, 2]);
        out_data.data = ArrayD::random(IxDyn(&[1, 2, 2]), Uniform::new(-5., 5.0));
        let input_data = vec![Rc::new(RefCell::new(input_data))];
        let out_data = vec![Rc::new(RefCell::new(out_data))];
        layer.forward_with_tensors(&input_data, &out_data).unwrap();

        info!("-------input_data-------");
        info!(" \n{:?}\n", input_data[0].as_ref().borrow().data);
        info!("-------output_data");
        info!("\n{:?}\n", out_data[0].as_ref().borrow().data);
    }
}
