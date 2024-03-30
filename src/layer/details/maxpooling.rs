use std::rc::Rc;

use crate::data::SharedTensor;
use crate::layer::abstract_layer::layer::*;
use crate::layer::abstract_layer::RuntimeOperatorData;
use crate::layer::Layer;
use crate::layer::ParameterData;

use log::error;
use log::info;
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
    A: Clone + Zero + PartialOrd + std::ops::Neg + Bounded + std::fmt::Debug + 'static,
    f32: From<A>,
    A: From<f32>,
{
    fn _new(
        padding_h: usize,
        padding_w: usize,
        pooling_size_h: usize,
        pooling_size_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Self {
        MaxPoolingLayer {
            runtime_operator: RuntimeOperatorData::new(),
            layer_name: "nn.MaxPool2d".to_string(),
            padding_h,
            padding_w,
            pooling_size_h,
            pooling_size_w,
            stride_h,
            stride_w,
        }
    }

    fn calculate_maxpool_window_value(
        &self,
        inputs: &SharedTensor<A>,
        cur_channel: usize,
        row_start: usize,
        col_start: usize,
    ) -> A {
        let input_h = inputs.as_ref().borrow().rows();
        let input_w = inputs.as_ref().borrow().cols();
        let mut max_val = A::min_value();
        for r in 0..self.pooling_size_h {
            for c in 0..self.pooling_size_w {
                let mut currect_val: A = Zero::zero();
                let cur_row = row_start + r;
                let cur_col = col_start + c;

                if cur_row >= self.padding_h
                    && cur_col >= self.padding_w
                    && cur_row < input_h + self.padding_h
                    && cur_col < input_w + self.padding_w
                {
                    currect_val = inputs
                        .as_ref()
                        .borrow()
                        .index(&[
                            cur_channel,
                            cur_row - self.padding_h,
                            cur_col - self.padding_w,
                        ])
                        .clone();
                }

                max_val = partial_min_max::max(max_val, currect_val);
            }
        }
        max_val
    }
    pub fn get_instance(
        runtime_operator: SharedRuntimeOperator<A>,
    ) -> Result<Rc<dyn Layer<A>>, LayerError> {
        let params_map = &runtime_operator.as_ref().borrow().params;
        //获取stride的参数
        let stride = ParameterData::get_stride(params_map)?;
        // 获取padding的参数
        let padding = ParameterData::get_padding(params_map)?;
        // 获取kernel_size
        let kernel_size = ParameterData::get_kernel_size(params_map)?;

        let maxpool_layer = MaxPoolingLayer {
            runtime_operator: RuntimeOperatorData::new_from(runtime_operator.clone()),
            layer_name: "nn.MaxPool2d".to_string(),
            padding_h: padding[0].try_into().unwrap(),
            padding_w: padding[1].try_into().unwrap(),
            pooling_size_h: kernel_size[0].try_into().unwrap(),
            pooling_size_w: kernel_size[1].try_into().unwrap(),
            stride_h: stride[0].try_into().unwrap(),
            stride_w: stride[1].try_into().unwrap(),
        };
        info!("MaxPoolingLayer 参数:");
        info!("Layer Name: {}", maxpool_layer.layer_name);
        info!(
            "Padding (H, W): ({}, {})",
            maxpool_layer.padding_h, maxpool_layer.padding_w
        );
        info!(
            "Pooling Size (H, W): ({}, {})",
            maxpool_layer.pooling_size_h, maxpool_layer.pooling_size_w
        );
        info!(
            "Stride (H, W): ({}, {})",
            maxpool_layer.stride_h, maxpool_layer.stride_w
        );
        Ok(Rc::new(maxpool_layer))
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
            error!(
                "now output data shape is {:?}, expect outdata shape is{:?} ",
                (outputs.borrow().rows(), outputs.borrow().cols()),
                (output_h, output_w)
            );

            return Err(LayerError::InferFailedOutputSizeError);
        }
        let input_padded_h = input_h + 2 * self.padding_h;
        let input_padded_w = input_w + 2 * self.padding_w;
        for channel_index in 0..input_c {
            for c in (0..=input_padded_w - self.pooling_size_w).step_by(self.stride_w) {
                for r in (0..=input_padded_h - self.pooling_size_h).step_by(self.stride_h) {
                    let max_val = self.calculate_maxpool_window_value(inputs, channel_index, r, c);

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
    A: Clone + Zero + std::ops::Neg + 'static + PartialOrd + Bounded + std::fmt::Debug,
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

    use super::MaxPoolingLayer;

    fn get_test_maxpooling_operator() -> SharedRuntimeOperator<f32> {
        let mut maxpooling_operator = RuntimeOperator::<f32>::new();
        maxpooling_operator.type_name = "nn.MaxPool2d".to_string();
        maxpooling_operator.params.insert(
            "stride".to_string(),
            Rc::new(RefCell::new(Parameter::IntList(vec![2, 2]))),
        );
        maxpooling_operator.params.insert(
            "kernel_size".to_string(),
            Rc::new(RefCell::new(Parameter::IntList(vec![2, 2]))),
        );
        maxpooling_operator.params.insert(
            "padding".to_string(),
            Rc::new(RefCell::new(Parameter::IntList(vec![0, 0]))),
        );

        Rc::new(RefCell::new(maxpooling_operator))
    }
    #[test]
    fn test_new_maxpooling() {
        let _maxpooling = MaxPoolingLayer::<f32>::_new(0, 0, 2, 2, 2, 2);
    }
    #[test]
    fn test_create_layer_find() {
        // 检查nn.MaxPool2d 算子是否注册
        let layer_type = "nn.MaxPool2d".to_string();
        assert!(LayerRegisterer::check_operator_registration(&layer_type));
    }
    #[test]
    fn test_create_layer_poolingforward() {
        let runtime_operator = get_test_maxpooling_operator();
        let _layer = LayerRegisterer::create_layer(&runtime_operator);
    }
    #[test_log::test]
    fn test_create_layer_poolingforward_1() {
        let runtime_operator = get_test_maxpooling_operator();
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
