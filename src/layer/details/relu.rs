use std::rc::Rc;

use num_traits::Zero;

use crate::data::SharedTensor;
use crate::layer::abstract_layer::layer::*;
use crate::layer::abstract_layer::RuntimeOperatorData;
use crate::layer::Layer;

use crate::runtime::SharedRuntimeOperator;

pub struct ReLULayer<A> {
    runtime_operator: RuntimeOperatorData<A>,
    layer_name: String,
}

impl<A> ReLULayer<A>
where
    A: Clone + Zero + PartialOrd + 'static,
{
    fn new(runtime_operator: RuntimeOperatorData<A>) -> Self {
        ReLULayer {
            runtime_operator: runtime_operator,
            layer_name: "nn.ReLu".to_string(),
        }
    }
    pub fn get_instance(runtime_operator: SharedRuntimeOperator<A>) -> Rc<dyn Layer<A>> {
        Rc::new(Self::new(RuntimeOperatorData::new_from(runtime_operator)))
    }
}
impl<A> Layer<A> for ReLULayer<A>
where
    A: Clone + Zero + PartialOrd,
{
    fn forward(&self) -> Result<(), crate::layer::abstract_layer::layer::LayerError> {
        let layer_input_datas = self.runtime_operator.prepare_input_tensor();
        let layer_ouput_datas = self.runtime_operator.prepare_output_tensor();

        self.forward_with_tensors(&layer_input_datas, &layer_ouput_datas)
            .unwrap();

        Ok(())
    }
    fn forward_with_tensors(
        &self,
        inputs: &Vec<SharedTensor<A>>,
        outputs: &Vec<SharedTensor<A>>,
    ) -> Result<(), crate::layer::abstract_layer::layer::LayerError> {
        if let Err(e) = self.check_inputs_and_outputs(inputs, outputs) {
            return Err(e);
        }
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
}
