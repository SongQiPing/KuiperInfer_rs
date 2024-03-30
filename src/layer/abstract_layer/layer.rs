use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use log::error;

use num_traits::Zero;

use crate::data::SharedTensor;
use crate::data::Tensor;
use crate::pnnx::Parameter;
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
    InferFailedOutputSizeError,
    InferFailedStrideParameterError,
    InferFailedBiasParameterError,

    ParameterMissingStrideError,
    ParameterMissingPaddingError,
    ParameterMissingKernelError,
    ParameterMissingDilationError,
    ParameterMissingInChannelError,
    ParameterMissingOutChannelError,
    ParameterMissingUseBiasError,
    ParameterMissingPaddingModeError,
    AttrMissingWeightError,
    AttrMissingBiasError,
}
pub trait Layer<A>
where
    A: Clone + Zero + std::fmt::Debug,
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
        if inputs.is_empty() {
            return Err(LayerError::InferFailedInputEmptyError);
        }
        // 检查输入输出张量是不是一样的
        if inputs.len() != outputs.len() {
            return Err(LayerError::InferFailedInputOutSizeMatchError);
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
        }
        Ok(())
    }

    fn layer_name(&self) -> &String;
}

pub struct ParameterData {}

impl ParameterData {
    fn get_int_vec_params(
        params_map: &HashMap<String, Rc<RefCell<Parameter>>>,
        params_name: &str,
    ) -> Option<Vec<i32>> {
        match params_map.get(&params_name.to_string()) {
            Some(params) => {
                let borrowed_params = params.borrow();
                match borrowed_params.clone() {
                    Parameter::IntList(stride) => Some(stride),

                    _ => None,
                }
            }
            None => None,
        }
    }
    fn get_int_params(
        params_map: &HashMap<String, Rc<RefCell<Parameter>>>,
        params_name: &str,
    ) -> Option<i32> {
        match params_map.get(&params_name.to_string()) {
            Some(params) => {
                let borrowed_params = params.borrow();
                match borrowed_params.get_int().clone() {
                    Some(val) => Some(val as i32),
                    None => None,
                }
            }
            None => None,
        }
    }
    fn get_bool_params(
        params_map: &HashMap<String, Rc<RefCell<Parameter>>>,
        params_name: &str,
    ) -> Option<bool> {
        match params_map.get(&params_name.to_string()) {
            Some(params) => {
                let borrowed_params = params.borrow();
                match borrowed_params.get_bool().clone() {
                    Some(val) => Some(val),
                    None => None,
                }
            }
            None => None,
        }
    }
    fn get_str_params(
        params_map: &HashMap<String, Rc<RefCell<Parameter>>>,
        params_name: &str,
    ) -> Option<String> {
        match params_map.get(&params_name.to_string()) {
            Some(params) => {
                let borrowed_params = params.borrow();
                match borrowed_params.get_string().clone() {
                    Some(val) => Some(val),
                    None => None,
                }
            }
            None => None,
        }
    }
    pub fn get_stride(
        params_map: &HashMap<String, Rc<RefCell<Parameter>>>,
    ) -> Result<Vec<i32>, LayerError> {
        // 获取值
        let stride = Self::get_int_vec_params(params_map, &"stride");
        if let None = stride {
            error!("Can not find the stride parameter");
            error!(
                "the params is {:?}",
                params_map.get(&"stride".to_string()).clone()
            );
            return Err(LayerError::ParameterMissingStrideError);
        }

        let stride = stride.unwrap();

        // 检查值是否符合要求
        if stride.len() == 2 {
            Ok(stride)
        } else {
            error!(
                "Can not find the right stride parameter, current stride parameter is {:?}",
                stride
            );
            Err(LayerError::ParameterMissingStrideError)
        }
    }
    pub fn get_padding(
        params_map: &HashMap<String, Rc<RefCell<Parameter>>>,
    ) -> Result<Vec<i32>, LayerError> {
        // 获取值
        let padding = Self::get_int_vec_params(params_map, &"padding");
        if let None = padding {
            error!("Can not find the padding parameter");
            error!(
                "the params is {:?}",
                params_map.get(&"padding".to_string()).clone()
            );
            return Err(LayerError::ParameterMissingPaddingError);
        }

        let padding = padding.unwrap();

        // 检查值是否符合要求
        if padding.len() == 2 {
            Ok(padding)
        } else {
            error!(
                "Can not find the right stride parameter, current stride parameter is {:?}",
                padding
            );
            Err(LayerError::ParameterMissingStrideError)
        }
    }
    pub fn get_kernel_size(
        params_map: &HashMap<String, Rc<RefCell<Parameter>>>,
    ) -> Result<Vec<i32>, LayerError> {
        // 获取值
        let kernel_size = Self::get_int_vec_params(params_map, &"kernel_size");
        if let None = kernel_size {
            error!("Can not find the kernel_size parameter");
            error!(
                "the params is {:?}",
                params_map.get(&"kernel_size".to_string()).clone()
            );
            return Err(LayerError::ParameterMissingKernelError);
        }

        let kernel_size = kernel_size.unwrap();

        // 检查值是否符合要求
        if kernel_size.len() == 2 {
            Ok(kernel_size)
        } else {
            error!(
                "Can not find the right stride parameter, current kernel_size parameter is {:?}",
                kernel_size
            );
            Err(LayerError::ParameterMissingKernelError)
        }
    }
    pub fn get_dilation(
        params_map: &HashMap<String, Rc<RefCell<Parameter>>>,
    ) -> Result<Vec<i32>, LayerError> {
        match Self::get_int_vec_params(params_map, &"dilation") {
            Some(stride) => {
                if stride.len() == 2 {
                    if stride[0] == 1 && stride[1] == 1 {
                        Ok(stride)
                    } else {
                        error!("Only support dilation value equals to one!, {:?}", stride);
                        Err(LayerError::ParameterMissingDilationError)
                    }
                } else {
                    error!("Can not find the dilation parameter!");
                    Err(LayerError::ParameterMissingDilationError)
                }
            }
            None => {
                error!("Can not find the dilation parameter");
                error!(
                    "the params is {:?}",
                    params_map.get(&"dilation".to_string()).clone()
                );
                Err(LayerError::ParameterMissingDilationError)
            }
        }
    }
    pub fn get_in_channels(
        params_map: &HashMap<String, Rc<RefCell<Parameter>>>,
    ) -> Result<i32, LayerError> {
        //获取值值
        let inchannels = Self::get_int_params(params_map, &"in_channels");
        if let None = inchannels {
            error!("Can not find the in_channels parameter");
            error!(
                "the params is {:?}",
                params_map.get(&"in_channels".to_string()).clone()
            );
            return Err(LayerError::ParameterMissingInChannelError);
        }
        let in_channels = inchannels.unwrap();

        Ok(in_channels)
    }
    pub fn get_out_channels(
        params_map: &HashMap<String, Rc<RefCell<Parameter>>>,
    ) -> Result<i32, LayerError> {
        match Self::get_int_params(params_map, &"out_channels") {
            Some(stride) => Ok(stride),
            None => {
                error!("Can not find the out_channels parameter");
                error!(
                    "the params is {:?}",
                    params_map.get(&"out_channels".to_string()).clone()
                );
                Err(LayerError::ParameterMissingOutChannelError)
            }
        }
    }
    pub fn get_groups(
        params_map: &HashMap<String, Rc<RefCell<Parameter>>>,
    ) -> Result<i32, LayerError> {
        match Self::get_int_params(params_map, &"groups") {
            Some(stride) => Ok(stride),
            None => {
                error!("Can not find the groups parameter");
                error!(
                    "the params is {:?}",
                    params_map.get(&"groups".to_string()).clone()
                );
                Err(LayerError::ParameterMissingOutChannelError)
            }
        }
    }
    pub fn get_has_bias(
        params_map: &HashMap<String, Rc<RefCell<Parameter>>>,
    ) -> Result<bool, LayerError> {
        match Self::get_bool_params(params_map, &"bias") {
            Some(has_bias) => Ok(has_bias),
            None => {
                error!("Can not find the bias parameter");
                error!(
                    "the params is {:?}",
                    params_map.get(&"bias".to_string()).clone()
                );
                Err(LayerError::ParameterMissingUseBiasError)
            }
        }
    }
    pub fn get_padding_mode(
        params_map: &HashMap<String, Rc<RefCell<Parameter>>>,
    ) -> Result<String, LayerError> {
        match Self::get_str_params(params_map, &"padding_mode") {
            Some(padding_mode) => {
                if padding_mode == "zeros".to_string() {
                    Ok(padding_mode)
                } else {
                    error!("Padding mode unsupported: {}", padding_mode);
                    Err(LayerError::ParameterMissingPaddingModeError)
                }
            }
            None => {
                error!("Can not find the padding_mode parameter");
                error!(
                    "the params is {:?}",
                    params_map.get(&"padding_mode".to_string()).clone()
                );
                Err(LayerError::ParameterMissingPaddingModeError)
            }
        }
    }
}
pub trait ParameteraGetterSetter<A> {
    fn set(&mut self, weights: &Vec<SharedTensor<A>>) -> Result<(), LayerError>;
    fn get(&self) -> Result<&Vec<SharedTensor<A>>, LayerError>;
}
pub struct TensorData<A> {
    data: Option<Vec<SharedTensor<A>>>,
}
impl<A> ParameteraGetterSetter<A> for TensorData<A> {
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
    pub fn new() -> Self {
        RuntimeOperatorData {
            runtime_operator: None,
        }
    }
    pub fn new_from(runtime_operator: SharedRuntimeOperator<A>) -> Self {
        RuntimeOperatorData {
            runtime_operator: Some(runtime_operator),
        }
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

pub trait ParamDataGetterSetter<A> {
    fn set(&mut self, weights: &SharedTensor<A>);
    fn get(&self) -> &SharedTensor<A>;
}

#[derive(Debug)]
pub struct WeightData<A>
where
    A: Clone + Zero,
{
    pub weights: SharedTensor<A>,
}
impl<A> WeightData<A>
where
    A: Clone + Zero,
{
    pub fn init_param(
        param_cout: usize,
        param_channel: usize,
        param_height: usize,
        param_width: usize,
    ) -> Self {
        let tensor = Tensor::<A>::new(&[param_cout, param_channel, param_height, param_width]);
        WeightData {
            weights: Rc::new(RefCell::new(tensor)),
        }
    }
}
impl<A> ParamDataGetterSetter<A> for WeightData<A>
where
    A: Clone + Zero,
{
    fn set(&mut self, weights: &SharedTensor<A>) {
        assert_eq!(
            self.weights.as_ref().borrow().batch_size(),
            weights.as_ref().borrow().batch_size()
        );
        assert_eq!(
            self.weights.as_ref().borrow().channels(),
            weights.as_ref().borrow().channels()
        );
        assert_eq!(
            self.weights.as_ref().borrow().rows(),
            weights.as_ref().borrow().rows()
        );
        assert_eq!(
            self.weights.as_ref().borrow().cols(),
            weights.as_ref().borrow().cols()
        );
        self.weights = weights.clone();
    }
    fn get(&self) -> &SharedTensor<A> {
        &self.weights
    }
}
#[cfg(test)]
mod test_abrastra_layer {

    #[test]
    fn test_forward() {}
}
