use std::rc::Rc;

use ndarray::LinalgScalar;
use num_traits::Bounded;
use num_traits::Zero;

use crate::data::SharedTensor;
use crate::layer::abstract_layer::layer::*;
use crate::layer::abstract_layer::RuntimeOperatorData;
use crate::layer::Layer;
use crate::layer::LayerError;
use crate::runtime::SharedRuntimeOperator;

pub struct ReLULayer<A> {
    runtime_operator: RuntimeOperatorData<A>,
    layer_name: String,
}

impl<A> ReLULayer<A>
where
    A: Clone
        + Zero
        + LinalgScalar
        + std::ops::Neg
        + PartialOrd
        + 'static
        + num_traits::Bounded
        + std::fmt::Debug,
{
    fn new(runtime_operator: RuntimeOperatorData<A>) -> Self {
        ReLULayer {
            runtime_operator: runtime_operator,
            layer_name: "nn.ReLu".to_string(),
        }
    }
    pub fn get_instance(
        runtime_operator: SharedRuntimeOperator<A>,
    ) -> Result<Rc<dyn Layer<A>>, LayerError> {
        Ok(Rc::new(Self::new(RuntimeOperatorData::new_from(
            runtime_operator,
        ))))
    }
}
impl<A> Layer<A> for ReLULayer<A>
where
    A: Clone + Zero + std::ops::Neg + 'static + PartialOrd + Bounded + std::fmt::Debug,
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
            if input_data.as_ref().borrow().empty() {
                return Err(LayerError::InferFailedInputEmptyError);
            }
            if input_data.borrow().channels() != output_data.borrow().channels() {
                return Err(LayerError::InferFailedInputOutSizeMatchError);
            }
            let input_ndarry_data = input_data.as_ref().borrow().data.mapv(|x| {
                if x < A::zero() {
                    A::zero()
                } else {
                    x.clone()
                }
            });
            output_data.borrow_mut().set_data(input_ndarry_data);
        }
        Ok(())
    }

    fn layer_name(&self) -> &String {
        &self.layer_name
    }
}

#[cfg(test)]
mod test_abrastra_layer {
    use std::{cell::RefCell, rc::Rc};

    use super::*;
    use crate::data::Tensor;
    use crate::layer::LayerRegisterer;
    use ndarray::ArrayD;
    use ndarray::IxDyn;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_relu_forward() {
        let input_tensor = Tensor::<f32>::new(&[16, 13, 15]);
        let input_data = vec![Rc::new(RefCell::new(input_tensor))];

        let output_tensor = Tensor::<f32>::new(&[16, 13, 15]);
        let output_data = vec![Rc::new(RefCell::new(output_tensor))];
        let runtime_operator = RuntimeOperatorData::<f32>::new();

        let relu_layer = ReLULayer {
            layer_name: "relu".to_string(),
            runtime_operator: runtime_operator,
        };
        relu_layer
            .forward_with_tensors(&input_data, &output_data)
            .unwrap();
    }
    #[test]
    fn test_create_layer_find() {
        // 检查nn.ReLu 算子是否注册
        let layer_type = "nn.ReLu".to_string();
        assert!(LayerRegisterer::check_operator_registration(&layer_type));
    }

    #[test_log::test]
    fn test_create_layer_reluforward() {
        use crate::layer::LayerRegisterer;
        use crate::runtime::RuntimeOperator;
        use log::info;
        use std::cell::RefCell;
        use std::rc::Rc;
        let mut runtime_operator = RuntimeOperator::<f32>::new();
        runtime_operator.type_name = "nn.ReLu".to_string();
        let runtime_operator = Rc::new(RefCell::new(runtime_operator));
        let relu_layer = LayerRegisterer::create_layer(&runtime_operator);

        let mut input_data = Tensor::<f32>::new(&[3, 4, 4]);
        input_data.data = ArrayD::random(IxDyn(&[3, 4, 4]), Uniform::new(-5., 5.0));

        info!("{:?}", input_data);
        let input_data = vec![Rc::new(RefCell::new(input_data))];
        let out_data = input_data.clone();
        relu_layer
            .forward_with_tensors(&input_data, &out_data)
            .unwrap();
        info!("{:?}", out_data);
    }
}
