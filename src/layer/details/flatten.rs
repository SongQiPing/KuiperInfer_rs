use std::rc::Rc;

use crate::data::SharedTensor;
use crate::layer::abstract_layer::layer::*;
use crate::layer::abstract_layer::RuntimeOperatorData;
use crate::layer::Layer;
use crate::layer::ParameterData;
use crate::TensorScalar;

use log::info;

use ndarray::IxDyn;

use crate::layer::LayerError;
use crate::runtime::SharedRuntimeOperator;

pub struct FlattenLayer<A> {
    runtime_operator: RuntimeOperatorData<A>,
    layer_name: String,
    start_dim: i32,
    end_dim: i32,
}

impl<A> FlattenLayer<A>
where
    A: TensorScalar + std::fmt::Debug,
    f32: From<A>,
    A: From<f32>,
{
    pub fn get_instance(
        runtime_operator: SharedRuntimeOperator<A>,
    ) -> Result<Rc<dyn Layer<A>>, LayerError> {
        let params_map = &runtime_operator.as_ref().borrow().params;
        //获取stride的参数
        let start_dim = ParameterData::get_start_dim(params_map)?;
        let end_dim = ParameterData::get_end_dim(params_map)?;

        let flatten_layer = FlattenLayer {
            runtime_operator: RuntimeOperatorData::new_from(runtime_operator.clone()),
            layer_name: "nn.AdaptiveAvgPool2d".to_string(),
            start_dim: start_dim.try_into().unwrap(),
            end_dim: end_dim.try_into().unwrap(),
        };

        info!("AdaptiveAveragePoolingLayer 参数:");
        info!("Layer Name: {}", flatten_layer.layer_name);
        info!("start_dim (): ({})", flatten_layer.start_dim);
        info!("end_dim (): ({})", flatten_layer.end_dim);
        Ok(Rc::new(flatten_layer))
    }

    fn batch_forward_wight_tensors(
        &self,
        inputs: &SharedTensor<A>,
        outputs: &SharedTensor<A>,
    ) -> Result<(), LayerError> {
        let mut start_dim = self.start_dim;
        let mut end_dim = self.end_dim;
        let total_dims = 4;
        if start_dim < 0 {
            start_dim = total_dims + start_dim;
        }
        if end_dim < 0 {
            end_dim = total_dims + end_dim;
        }
        assert!(
            end_dim > start_dim,
            "The end dim must greater than start dim"
        );
        assert!(
            end_dim <= 3 && start_dim >= 1,
            "The end dim must less than two and start dim must greater than zero"
        );
        let mut input_shape = inputs.as_ref().borrow().shapes().clone();
        input_shape.insert(0, 1);
        let elements_size = input_shape[start_dim as usize..=end_dim as usize]
            .iter()
            .fold(1, |acc, &x| acc * x);

        let input_data = inputs.as_ref().borrow().data().clone();

        if start_dim == 1 && end_dim == 3 {
            let output_shape = IxDyn(&[elements_size]);
            let out_data = input_data.into_shape(output_shape).unwrap();
            outputs.as_ref().borrow_mut().set_data(out_data);
        } else if start_dim == 2 && end_dim == 3 {
            let channels = inputs.as_ref().borrow().channels();
            let output_shape = IxDyn(&[channels, elements_size]);
            let out_data = input_data.into_shape(output_shape).unwrap();
            outputs.as_ref().borrow_mut().set_data(out_data);
        } else if start_dim == 1 && end_dim == 2 {
            let cols = inputs.as_ref().borrow().cols();
            let output_shape = IxDyn(&[elements_size, cols]);
            let out_data = input_data.into_shape(output_shape).unwrap();
            outputs.as_ref().borrow_mut().set_data(out_data);
        } else {
            panic!(
                "Wrong flatten dim: start dim: {} end dim: {}",
                start_dim, end_dim
            );
        }

        Ok(())
    }
}
use std::convert::From;

impl<A> Layer<A> for FlattenLayer<A>
where
    A: TensorScalar + std::fmt::Debug,
    f32: From<A>,
    A: From<f32>,
{
    fn forward(&self) -> Result<(), LayerError> {
        let layer_input_datas = self.runtime_operator.prepare_input_tensor();
        let layer_ouput_datas = self.runtime_operator.prepare_output_tensor();

        // if let Err(e) = self.check_inputs_and_outputs(&layer_input_datas, &layer_ouput_datas) {
        //     return Err(e);
        // }

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

    fn get_test_flatten_operator() -> SharedRuntimeOperator<f32> {
        let mut flatten_operator = RuntimeOperator::<f32>::new();
        flatten_operator.type_name = "torch.flatten".to_string();
        flatten_operator.params.insert(
            "start_dim".to_string(),
            Rc::new(RefCell::new(Parameter::Int(1))),
        );
        flatten_operator.params.insert(
            "end_dim".to_string(),
            Rc::new(RefCell::new(Parameter::Int(-1))),
        );
        Rc::new(RefCell::new(flatten_operator))
    }

    #[test]
    fn test_create_layer_find() {
        // 检查nn.MaxPool2d 算子是否注册
        let layer_type = "torch.flatten".to_string();
        assert!(LayerRegisterer::check_operator_registration(&layer_type));
    }
    #[test]
    fn test_create_layer_poolingforward() {
        let runtime_operator = get_test_flatten_operator();
        let _layer = LayerRegisterer::create_layer(&runtime_operator);
    }
    #[test_log::test]
    fn test_create_layer_poolingforward_1() {
        let runtime_operator = get_test_flatten_operator();
        let layer = LayerRegisterer::create_layer(&runtime_operator);

        let mut input_data = Tensor::<f32>::new(&[512, 1, 1]);
        input_data.data = ArrayD::random(IxDyn(&[512, 1, 1]), Uniform::new(-5., 5.0));
        let mut out_data = Tensor::<f32>::new(&[512]);
        out_data.data = ArrayD::random(IxDyn(&[512]), Uniform::new(-5., 5.0));
        let input_data = vec![Rc::new(RefCell::new(input_data))];
        let out_data = vec![Rc::new(RefCell::new(out_data))];
        layer.forward_with_tensors(&input_data, &out_data).unwrap();

        info!("-------input_data-------");
        info!(" \n{:?}\n", input_data[0].as_ref().borrow().data);
        info!("-------output_data");
        info!("\n{:?}\n", out_data[0].as_ref().borrow().data);
    }
}
