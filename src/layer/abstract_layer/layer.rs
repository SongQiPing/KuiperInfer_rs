
use num_traits::Zero;

use crate::data::SharedTensor;
use crate::runtime::SharedRuntimeOperator;
#[derive(Debug)]
pub enum LayerError {
    // LocalFileHeaderInvalid,
    // CentralDirectoryFileHeaderInvalid,
    // EndOfCentralDirectoryInvalid,
    // CantSeekToDirEnd,
    // CantSeekToFileHeader,
    // CantSeekSkip,
    // ParseError,
    // ReadStringError,
    // CantSeekToDirStart,
    // UnsupportedCompressionMethod,
    // DecompressionError,
    // DataReadError,
    NoWeightsAvailableError,
    NoBiasAvailableError,
    NoRuntimeOperatorAvailableError,

    InferFailedInputEmptyError,
    InferFailedInputOutSizeMatchError,
    
}
pub trait Layer<A> 
where 
    A:Zero+Clone
{
    fn forward(&self) -> Result<(), LayerError>;
    fn forward_with_tensors(
        &self,
        inputs: &Vec<SharedTensor<A>>,
        outputs: &Vec<SharedTensor<A>>,
    ) -> Result<(), LayerError>;
    fn check_inputs_and_outputs(
        &self,
        inputs: &Vec<SharedTensor<A>>,
        outputs: &Vec<SharedTensor<A>>,
    ) -> Result<(), LayerError> {
        // 检查输入的张量是否为空
        if inputs.is_empty(){
            return Err(LayerError::InferFailedInputEmptyError);
        }
        // 检查输入输出张量是不是一样的
        if inputs.len() != outputs.len(){
            return Err(LayerError::InferFailedInputOutSizeMatchError);
        }

        let batch_size = inputs.len();
        for i in 0..batch_size{
            let input_data = &inputs[i];
            let output_data = &outputs[i];
            if input_data.as_ref().borrow().empty(){
                return Err(LayerError::InferFailedInputEmptyError);
            }
            if input_data.borrow().channels() != output_data.borrow().channels(){
                return Err(LayerError::InferFailedInputOutSizeMatchError);
            }
        }
        Ok(())

    }

    fn layer_name(&self) -> &String;
}

pub trait ParameteraGetterSetter<A> {
    fn set(&mut self, weights: &Vec<SharedTensor<A>>) -> Result<(), LayerError>;
    fn get(&self) -> Result<&Vec<SharedTensor<A>>, LayerError>;
}
pub struct ParameterData<A> {
    data: Option<Vec<SharedTensor<A>>>,
}
impl<A> ParameteraGetterSetter<A> for ParameterData<A> {
    fn get(&self) -> Result<&Vec<SharedTensor<A>>, LayerError> {
        match &self.data {
            Some(data) => Ok(&data),
            None => Err(LayerError::NoWeightsAvailableError),
        }
    }
    fn set(&mut self, weights: &Vec<SharedTensor<A>>) -> Result<(), LayerError> {
        self.data = Some(weights.clone());
        Ok(())
    }
}
pub trait RuntimeOperatorGetterSetter<A> {
    fn set(&mut self, runtime_operatpr: &SharedRuntimeOperator<A>);
    fn get(&self) -> Result<&SharedRuntimeOperator<A>, LayerError>;
    fn prepare_input_tensor(&self) -> Vec<SharedTensor<A>>;
    fn prepare_output_tensor(&self) -> Vec<SharedTensor<A>>;
}
pub struct RuntimeOperatorData<A> {
    runtime_operator: Option<SharedRuntimeOperator<A>>,
}

impl<A> RuntimeOperatorData<A> {
    pub fn new() -> Self{
        RuntimeOperatorData { runtime_operator: None }
    }
    pub fn new_from(runtime_operator:SharedRuntimeOperator<A>) -> Self{
        RuntimeOperatorData { runtime_operator: Some(runtime_operator) }
    }
}
impl<A> RuntimeOperatorGetterSetter<A> for RuntimeOperatorData<A> {
    fn get(&self) -> Result<&SharedRuntimeOperator<A>, LayerError> {
        match &self.runtime_operator {
            Some(runtime_operator) => Ok(&runtime_operator),
            None => Err(LayerError::NoRuntimeOperatorAvailableError),
        }
    }
    fn set(&mut self, runtime_operatpr: &SharedRuntimeOperator<A>) {
        self.runtime_operator = Some(runtime_operatpr.clone());
    }
    fn prepare_input_tensor(&self) -> Vec<SharedTensor<A>> {
        let runtime_operator = self.get().expect("RuntimeOperator not initialized");

        let intput_operand_datas = &runtime_operator.borrow().input_operands_seq;
        let mut layer_input_datas: Vec<SharedTensor<A>> = Vec::new();
        for intput_operand_data in intput_operand_datas {
            for input_data in &intput_operand_data.as_ref().borrow().datas {
                layer_input_datas.push(input_data.clone());
            }
        }

        layer_input_datas
    }
    fn prepare_output_tensor(&self) -> Vec<SharedTensor<A>> {
        let runtime_operator = self.get().expect("RuntimeOperator not initialized");

        // let output_operand_datas = &runtime_operator.borrow().output_operands.as_ref().unwrap();
        match &runtime_operator.borrow().output_operands {
            Some(output_operand_datas) => output_operand_datas.as_ref().borrow().datas.clone(),
            None => {
                panic!("RuntimeOperator not initialized")
            }
        }
    }
}

#[cfg(test)]
mod test_abrastra_layer {
 
    #[test]
    fn test_forward() {}
}
