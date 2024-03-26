use std::rc::Rc;

use num_traits::Zero;

use crate::data::SharedTensor;
use crate::layer::abstract_layer::layer::*;
use crate::layer::abstract_layer::RuntimeOperatorData;
use crate::layer::Layer;

use crate::runtime::SharedRuntimeOperator;

pub struct SigmoidLayer<A> {
    runtime_operator: RuntimeOperatorData<A>,
    layer_name: String,
}

impl<A> SigmoidLayer<A>
where
    A: Clone + Zero + std::ops::Neg + 'static,
    f32: From<A>,
    A: From<f32>,
{
    fn new(runtime_operator: RuntimeOperatorData<A>) -> Self {
        SigmoidLayer {
            runtime_operator: runtime_operator,
            layer_name: "nn.ReLu".to_string(),
        }
    }
    pub fn get_instance(runtime_operator: SharedRuntimeOperator<A>) -> Rc<dyn Layer<A>> {
        Rc::new(Self::new(RuntimeOperatorData::new_from(runtime_operator)))
    }
}
use std::convert::From;

impl<A> Layer<A> for SigmoidLayer<A>
where
    A: Clone + Zero + std::ops::Neg,
    A: From<f32>,
    f32: From<A>,
{
    fn forward(&self) -> Result<(), LayerError> {
        let layer_input_datas = self.runtime_operator.prepare_input_tensor();
        let layer_ouput_datas = self.runtime_operator.prepare_output_tensor();

        if let Err(e) =  self.check_inputs_and_outputs(&layer_input_datas, &layer_ouput_datas){
            return Err(e);
        }

        if let Err(e) =  self.forward_with_tensors(&layer_input_datas, &layer_ouput_datas){
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
                let mut x = f32::from(x);
                x = 1.0 / (1.0 + (-x).exp());

                let x = A::from(x);
                x
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
        let out_data =  vec![Rc::new(RefCell::new(out_data))];
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
