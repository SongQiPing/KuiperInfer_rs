use std::rc::Rc;

use num_traits::Bounded;
use num_traits::Zero;

use crate::data::SharedTensor;
use crate::layer::abstract_layer::layer::*;
use crate::layer::abstract_layer::RuntimeOperatorData;
use crate::layer::Layer;
use crate::TensorScalar;

use crate::runtime::SharedRuntimeOperator;

pub struct SoftmaxLayer<A> {
    runtime_operator: RuntimeOperatorData<A>,
    layer_name: String,
    softmax_dim: i32,
}

impl<A> SoftmaxLayer<A>
where
    A: Clone
        + Zero
        + std::ops::Neg
        + 'static
        + PartialOrd
        + Bounded
        + std::fmt::Debug
        + TensorScalar,
    f32: From<A>,
    A: From<f32>,
{
    pub fn new(softmax_dim: i32) -> Self {
        SoftmaxLayer {
            runtime_operator: RuntimeOperatorData::new(),
            layer_name: "nn.ReLu".to_string(),
            softmax_dim: softmax_dim,
        }
    }
    pub fn get_instance(
        runtime_operator: SharedRuntimeOperator<A>,
    ) -> Result<Rc<dyn Layer<A>>, LayerError> {
        //获取dim的参数
        let params_map = &runtime_operator.as_ref().borrow().params;
        //获取dim的参数
        let dim = ParameterData::get_dim(params_map)?;
        let softmax = SoftmaxLayer {
            runtime_operator: RuntimeOperatorData::new_from(runtime_operator.clone()),
            layer_name: "nn.ReLu".to_string(),
            softmax_dim: dim.try_into().unwrap(),
        };

        Ok(Rc::new(softmax))
    }
    fn pos_to_index(pos_id: usize, raw_shapes: Vec<usize>) -> Vec<usize> {
        let mut indices = Vec::new();
        let mut remainder = pos_id;
        let mut divisors = Vec::new();
        let mut product = 1;

        for shape in raw_shapes.iter().rev() {
            divisors.push(product);
            product *= shape;
        }

        for _ in raw_shapes.iter().rev() {
            let divisor = divisors.pop().unwrap();
            let quotient = remainder / divisor;
            let remainder_new = remainder % divisor;
            indices.push(quotient);
            remainder = remainder_new;
        }

        indices.reverse();
        indices
    }
    fn batch_forward_wight_tensors(
        &self,
        inputs: &SharedTensor<A>,
        outputs: &SharedTensor<A>,
    ) -> Result<(), LayerError> {
        let mut raw_shapes = inputs.as_ref().borrow().shapes().clone();
        let mut dim = self.softmax_dim;
        if dim < 0 {
            dim += raw_shapes.len() as i32;
        }
        if dim < 0 || dim >= 3 || dim > raw_shapes.len() as i32 {
            panic!(
                "Error softmax dimension, which need between 0 and 2, but dimension is {}",
                &dim
            );
        }
        let dim = dim as usize;
        let padding_size_num = 3 - raw_shapes.len();
        for _ in 0..padding_size_num {
            raw_shapes.push(1);
        }

        let mut outer_sizes: usize = 1;
        for i in 0..dim {
            outer_sizes = outer_sizes * raw_shapes[i];
        }

        let mut inner_sizes: usize = 1;
        for i in dim + 1..raw_shapes.len() {
            inner_sizes = inner_sizes * raw_shapes[i];
        }
        let axis_sizes = raw_shapes[dim];

        assert_eq!(
            axis_sizes * outer_sizes * inner_sizes,
            inputs.as_ref().borrow().size()
        );
        println!(
            "outer_sizes:{}, axis_sizes:{}, inner_sizes:{}",
            outer_sizes, axis_sizes, inner_sizes
        );
        let axis_sizes = raw_shapes[dim];
        for outer_size in 0..outer_sizes {
            for inner_size in 0..inner_sizes {
                let mut max_value = A::min_value();
                for axis_size in 0..axis_sizes {
                    let pos_index = outer_size * axis_sizes * inner_sizes
                        + axis_size * inner_sizes
                        + inner_size;
                    let index =
                        Self::pos_to_index(pos_index, inputs.as_ref().borrow().shapes().clone());
                    let cur_val = inputs.as_ref().borrow().index(&index).clone();
                    if cur_val > max_value {
                        max_value = cur_val;
                    }
                }
                let mut sum_value: A = A::zero();
                for axis_size in 0..axis_sizes {
                    let pos_index = outer_size * axis_sizes * inner_sizes
                        + axis_size * inner_sizes
                        + inner_size;
                    let index =
                        Self::pos_to_index(pos_index, inputs.as_ref().borrow().shapes().clone());
                    let cur_val = inputs.as_ref().borrow().index(&index).clone();
                    let x = f32::from(cur_val - max_value);

                    let exp_sub_value = A::from(x.exp());

                    sum_value = sum_value + exp_sub_value;
                    *outputs.as_ref().borrow_mut().index_mut(&index) = exp_sub_value;
                }
                // 迭代当前dim中的数据，求exp(cur_value - max_value) / sum_value
                for axis_size in 0..axis_sizes {
                    let pos_index = outer_size * axis_sizes * inner_sizes
                        + axis_size * inner_sizes
                        + inner_size;
                    let index =
                        Self::pos_to_index(pos_index, inputs.as_ref().borrow().shapes().clone());

                    let exp_sub_value = outputs.as_ref().borrow().index(&index).clone();

                    *outputs.as_ref().borrow_mut().index_mut(&index) = exp_sub_value / sum_value;
                }
            }
        }
        Ok(())
    }
}
use std::convert::From;

impl<A> Layer<A> for SoftmaxLayer<A>
where
    A: Clone
        + Zero
        + std::ops::Neg
        + 'static
        + PartialOrd
        + Bounded
        + std::fmt::Debug
        + TensorScalar,
    A: From<f32>,
    f32: From<A>,
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
mod test_softmax_layer {

    use std::cell::RefCell;
    use std::rc::Rc;

    use crate::data::Tensor;
    use crate::layer::LayerRegisterer;
    use crate::pnnx::Parameter;
    use crate::runtime::RuntimeOperator;
    use crate::runtime::SharedRuntimeOperator;
    use ndarray::ArrayD;
    use ndarray::IxDyn;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    fn get_test_softmax_operator() -> SharedRuntimeOperator<f32> {
        let mut maxpooling_operator = RuntimeOperator::<f32>::new();
        maxpooling_operator.type_name = "nn.Softmax".to_string();

        maxpooling_operator
            .params
            .insert("dim".to_string(), Rc::new(RefCell::new(Parameter::Int(-1))));

        Rc::new(RefCell::new(maxpooling_operator))
    }
    #[test]
    fn test_create_layer_find() {
        // 检查nn.ReLu 算子是否注册
        let layer_type = "nn.Softmax".to_string();
        assert!(LayerRegisterer::check_operator_registration(&layer_type));
    }

    #[test_log::test]
    fn test_create_layer_reluforward() {
        use crate::layer::LayerRegisterer;

        use log::info;
        use std::cell::RefCell;
        use std::rc::Rc;
        let runtime_operator = get_test_softmax_operator();

        let sigmoid_layer = LayerRegisterer::create_layer(&runtime_operator);

        let mut input_data = Tensor::<f32>::new(&[1000]);
        let out_data = Tensor::<f32>::new(&[1000]);
        input_data.data = ArrayD::random(IxDyn(&[1000]), Uniform::new(-5., 5.0));

        let input_data = vec![Rc::new(RefCell::new(input_data))];
        let out_data = vec![Rc::new(RefCell::new(out_data))];
        sigmoid_layer
            .forward_with_tensors(&input_data, &out_data)
            .unwrap();
        info!("-------input_data-------");
        info!("{:?}", input_data);
        info!("-------output_data");
        info!("{:?}", out_data);
    }
}
