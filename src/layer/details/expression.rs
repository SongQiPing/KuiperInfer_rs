use crate::data::torch_utils::tensor_element_add;
use crate::data::torch_utils::tensor_element_multiply;

use crate::layer::abstract_layer::layer::*;
use crate::layer::abstract_layer::RuntimeOperatorData;

use crate::parser::TokenType;
use crate::{layer::Layer, parser::ExpressionParser};
use crate::{layer::LayerError, pnnx::Parameter, runtime::SharedRuntimeOperator};
use log::error;

use ndarray::LinalgScalar;
use num_traits::Bounded;
use std::rc::Rc;

pub struct ExpressionLayer<A> {
    runtime_operator: RuntimeOperatorData<A>,
    layer_name: String,
    _statement: String,
    pub parser: Box<ExpressionParser>,
}
impl<A> ExpressionLayer<A>
where
    A: Clone + LinalgScalar + PartialOrd + Bounded + std::ops::Neg + std::fmt::Debug,
    f32: From<A>,
    A: From<f32>,
{
    pub fn new(runtime_operator: SharedRuntimeOperator<A>, statemen: String) -> ExpressionLayer<A> {
        let parser = ExpressionParser::from_string(&statemen);
        ExpressionLayer {
            runtime_operator: RuntimeOperatorData::new_from(runtime_operator.clone()),
            layer_name: "ExpressionLayer".to_string(),
            _statement: statemen.clone(),
            parser: Box::new(parser),
        }
    }
    pub fn new_from(statemen: String) -> ExpressionLayer<A> {
        let mut parser = ExpressionParser::from_string(&statemen);
        parser.tokenizer();
        ExpressionLayer {
            runtime_operator: RuntimeOperatorData::new(),
            layer_name: "ExpressionLayer".to_string(),
            _statement: statemen.clone(),
            parser: Box::new(parser),
        }
    }
    pub fn get_instance(
        runtime_operator: SharedRuntimeOperator<A>,
    ) -> Result<Rc<dyn Layer<A>>, LayerError> {
        if let None = runtime_operator
            .as_ref()
            .borrow()
            .params
            .get(&"expr".to_string())
        {
            return Err(LayerError::ParameterMissingExpr);
        }
        let param_map = &runtime_operator.as_ref().borrow();

        let statemen_param: &Rc<std::cell::RefCell<Parameter>> =
            param_map.params.get(&"expr".to_string()).unwrap();

        let statement = match statemen_param.borrow().to_owned() {
            Parameter::String(statement_string) => statement_string,
            _ => {
                error!("Can not find the expression parameter");
                return Err(LayerError::ParameterMissingExpr);
            }
        };
        let mut expressiong_layer = ExpressionLayer::new(runtime_operator.clone(), statement);
        expressiong_layer.parser.tokenizer();
        Ok(Rc::new(expressiong_layer))
    }
}

impl<A> Layer<A> for ExpressionLayer<A>
where
    A: Clone + LinalgScalar + PartialOrd + Bounded + std::ops::Neg + std::fmt::Debug,
    A: From<f32>,
    f32: From<A>,
{
    fn forward(&self) -> Result<(), crate::layer::LayerError> {
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
        inputs: &Vec<crate::data::SharedTensor<A>>,
        outputs: &Vec<crate::data::SharedTensor<A>>,
    ) -> Result<(), crate::layer::LayerError> {
        let token_nodes = self.parser.generate();
        let mut op_stack = Vec::new();
        let batch_size = outputs.len();

        for token_node in token_nodes {
            if let TokenType::TokenInputNumber(num_index) = token_node.as_ref().num_index {
                let mut input_token_nodes = Vec::new();
                let start_pos = num_index * batch_size;
                for i in 0..batch_size {
                    input_token_nodes.push(inputs.get(start_pos + i).unwrap().clone());
                }
                op_stack.push(input_token_nodes);
            } else {
                match token_node.as_ref().num_index {
                    TokenType::TokenAdd | TokenType::TokenMul => {}
                    _ => {
                        error!("Unknown operator type:{:?}", token_node.as_ref().num_index);
                    }
                }
                assert!(
                    op_stack.len() >= 2,
                    "The number of operand is less than two"
                );

                let input_node1 = op_stack.pop().unwrap();
                assert!(
                    input_node1.len() == batch_size,
                    "The first operand doesn't have appropriate number of tensors, which need{:?}",
                    batch_size
                );

                let input_node2 = op_stack.pop().unwrap();
                assert!(
                    input_node2.len() == batch_size,
                    "The first operand doesn't have appropriate number of tensors, which need{:?}",
                    batch_size
                );

                let mut output_token_nodes = Vec::new();
                for i in 0..batch_size {
                    //计算公式
                    match token_node.as_ref().num_index {
                        TokenType::TokenAdd => {
                            let tensor1 = input_node1.get(i).unwrap();
                            let tensor2 = input_node2.get(i).unwrap();
                            output_token_nodes.push(tensor_element_add(tensor1, tensor2));
                        }
                        TokenType::TokenMul => {
                            let tensor1 = input_node1.get(i).unwrap();
                            let tensor2 = input_node2.get(i).unwrap();
                            output_token_nodes.push(tensor_element_multiply(tensor1, tensor2));
                        }
                        _ => {
                            error!("Unknown operator type: {:?}", token_node.as_ref().num_index);
                        }
                    }
                }
                op_stack.push(output_token_nodes);
            }
        }
        assert!(
            op_stack.len() == 1,
            "The expression has more than one output operand!"
        );

        let out_node = op_stack.pop().unwrap();
        for i in 0..batch_size {
            outputs[i]
                .borrow_mut()
                .set_data(out_node[i].as_ref().borrow().data().clone())
        }
        Ok(())
    }
    fn layer_name(&self) -> &String {
        &self.layer_name
    }
}
#[cfg(test)]
mod test_expression_layer {
    use crate::Tensor;

    use super::*;
    #[test]
    fn tet_expression_layer() {
        let statement = "add(mul(@0,@1),@2)".to_string();

        let input1 = Tensor::<f32>::new(&[3, 224, 224]).shared_self();
        let input2 = Tensor::<f32>::new(&[3, 224, 224]).shared_self();
        let input3 = Tensor::<f32>::new(&[3, 224, 224]).shared_self();
        let inputs = vec![input1, input2, input3];
        let output = Tensor::<f32>::new(&[3, 224, 224]).shared_self();
        let outputs = vec![output];
        let layer = ExpressionLayer::<f32>::new_from(statement);
        layer.forward_with_tensors(&inputs, &outputs).unwrap();
    }
}
