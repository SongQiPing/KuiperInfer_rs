use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::data::SharedTensor;
use crate::layer::abstract_layer::layer::*;
use crate::layer::abstract_layer::RuntimeOperatorData;
use crate::layer::Layer;
use crate::layer::LayerError;
use crate::layer::WeightData;
use crate::runtime::RuntimeAttribute;
use crate::runtime::SharedRuntimeOperator;

use log::error;
use log::info;
use ndarray::prelude::*;
use ndarray::LinalgScalar;
use num_traits::Bounded;
use num_traits::Zero;

pub struct LinearLayer<A>
where
    A: Clone + Zero,
{
    runtime_operator: RuntimeOperatorData<A>,
    layer_name: String,
    in_features: usize,
    out_features: usize,
    use_bias: bool,
    weights: WeightData<A>,
    bias: Option<WeightData<A>>,
}

impl<A> LinearLayer<A>
where
    A: Clone
        + Zero
        + PartialOrd
        + std::ops::Neg
        + 'static
        + Bounded
        + std::marker::Copy
        + std::fmt::Debug
        + LinalgScalar,
    f32: From<A>,
    A: From<f32>,
{
    fn new(
        runtime_operator: Option<RuntimeOperatorData<A>>,
        in_features: usize,
        out_features: usize,
        use_bias: bool,
    ) -> Self {
        let runtime_operator = match runtime_operator {
            Some(runtime_operator) => runtime_operator,
            None => RuntimeOperatorData::new(),
        };

        let weight_data = WeightData::<A>::init_param(1, 1, out_features, in_features);

        let bias_data = match use_bias {
            true => Some(WeightData::<A>::init_param(1, 1, 1, out_features)),
            false => None,
        };
        // let kernel_matrix = Tensor::<A>::new(&[output_channel, in_channel * kernel_h * kernel_w]);

        let linear_layer = LinearLayer {
            runtime_operator: runtime_operator,
            layer_name: "nn.Linear".to_string(),
            in_features: in_features,
            out_features: out_features,
            use_bias: use_bias,
            weights: weight_data,
            bias: bias_data,
        };

        linear_layer
    }

    pub fn load_weight(
        &mut self,
        attribute_map: &HashMap<String, Rc<RefCell<RuntimeAttribute>>>,
    ) -> Result<(), LayerError> {
        match attribute_map.get(&"weight".to_string()) {
            Some(weight_attibute) => {
                let vec_data: Vec<A> = weight_attibute.as_ref().borrow_mut().get::<A>(false);
                let shape: [usize; 4] = [1, 1, self.out_features, self.in_features];
                let raw_shapes = IxDyn(&shape);
                let ndarray_data = ArrayD::from_shape_vec(raw_shapes, vec_data).unwrap();
                self.weights
                    .get()
                    .as_ref()
                    .borrow_mut()
                    .set_data(ndarray_data);
            }
            None => {
                error!("Can not find the weight attribute");
                return Err(LayerError::AttrMissingWeightError);
            }
        }
        Ok(())
    }
    pub fn load_bais(
        &mut self,
        attribute_map: &HashMap<String, Rc<RefCell<RuntimeAttribute>>>,
    ) -> Result<(), LayerError> {
        match attribute_map.get(&"bias".to_string()) {
            Some(weight_attibute) => {
                let vec_data: Vec<A> = weight_attibute.as_ref().borrow_mut().get::<A>(false);
                let shape: [usize; 4] = [1, 1, 1, self.out_features];
                let raw_shapes = IxDyn(&shape);
                let ndarray_data = ArrayD::from_shape_vec(raw_shapes, vec_data).unwrap();
                self.bias
                    .as_ref()
                    .unwrap()
                    .get()
                    .as_ref()
                    .borrow_mut()
                    .set_data(ndarray_data);
                info!("{:?}", self.bias);
            }
            None => {
                error!("Can not find the bias attribute");
                return Err(LayerError::AttrMissingBiasError);
            }
        }
        Ok(())
    }
    pub fn load_attribute(
        &mut self,
        attribute_map: &HashMap<String, Rc<RefCell<RuntimeAttribute>>>,
    ) -> Result<(), LayerError> {
        // 加载权重
        self.load_weight(attribute_map)?;
        // 加载偏置

        if self.use_bias {
            info!("加载偏置:{}", &self.use_bias);
            self.load_bais(attribute_map)?;
        }

        Ok(())
    }

    pub fn get_instance(
        runtime_operator: SharedRuntimeOperator<A>,
    ) -> Result<Rc<dyn Layer<A>>, LayerError> {
        let params_map = &runtime_operator.as_ref().borrow().params;

        let in_features = ParameterData::get_in_features(params_map)?;
        let out_features = ParameterData::get_out_features(params_map)?;

        // 获取bias
        let has_bias = ParameterData::get_has_bias(params_map)?;
        let mut conv_2d_layer = LinearLayer::<A>::new(
            Some(RuntimeOperatorData::new_from(runtime_operator.clone())),
            in_features.try_into().unwrap(),
            out_features.try_into().unwrap(),
            has_bias,
        );

        //加载权重
        conv_2d_layer.load_attribute(&runtime_operator.as_ref().borrow().attribute)?;

        conv_2d_layer.log_info();

        Ok(Rc::new(conv_2d_layer))
    }
    fn log_info(&self) {
        info!("ConvolutionLayer 参数:");
        info!("Layer Name: {}", self.layer_name);

        info!("in_features (): ({})", self.in_features);
        info!("out_feature (): ({})", self.out_features);
    }

    fn batch_forward_wight_tensors(
        &self,
        inputs: &SharedTensor<A>,
        outputs: &SharedTensor<A>,
    ) -> Result<(), LayerError> {
        let input_matrix = inputs.as_ref().borrow().data().clone();
        let input_matrix: ArrayBase<ndarray::OwnedRepr<A>, Dim<[usize; 2]>> =
            input_matrix.clone().into_dimensionality().unwrap();

        let weight = self.weights.get().as_ref().borrow().data().clone();
        let weight_shape = IxDyn(&[self.out_features, self.in_features]);
        let weight = weight.into_shape(weight_shape).unwrap();
        let weight: ArrayBase<ndarray::OwnedRepr<A>, Dim<[usize; 2]>> =
            weight.clone().into_dimensionality().unwrap();
        let weight = weight.t().to_owned();
        let mut output_tensor: ndarray::ArrayBase<ndarray::OwnedRepr<A>, ndarray::Dim<[usize; 2]>> =
            input_matrix.dot(&weight);
        if let Some(bias_data) = &self.bias {
            let bias_data = bias_data.get().borrow().data().clone();

            let bias_data = bias_data
                .into_shape(IxDyn(&[1, self.out_features]))
                .unwrap();
            let bias_data: ArrayBase<ndarray::OwnedRepr<A>, Dim<[usize; 2]>> =
                bias_data.clone().into_dimensionality().unwrap();

            output_tensor = output_tensor + bias_data;
        }
        let output_tensor = output_tensor.into_dyn();

        outputs.as_ref().borrow_mut().set_data(output_tensor);
        Ok(())
    }
    //     // /// 计算运算之后的形状
}
use std::convert::From;

impl<A> Layer<A> for LinearLayer<A>
where
    A: Clone + LinalgScalar + PartialOrd + std::ops::Neg + Bounded + std::fmt::Debug,
    f32: From<A>,
    A: From<f32>,
{
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

        Ok(())
    }
    fn forward(&self) -> Result<(), LayerError> {
        let layer_input_datas = self.runtime_operator.prepare_input_tensor();
        let layer_ouput_datas = self.runtime_operator.prepare_output_tensor();

        if let Err(e) = self.check_inputs_and_outputs(&layer_input_datas, &layer_ouput_datas) {
            return Err(e);
        }
        // 检查偏置是否存在
        if self.use_bias {
            if let None = self.bias {
                error!("The number of kernel matrix in the convolution layer should be greater than zero");
                return Err(LayerError::InferFailedBiasParameterError);
            }
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
            self.batch_forward_wight_tensors(input_data, output_data)?;
        }
        Ok(())
    }

    fn layer_name(&self) -> &String {
        &self.layer_name
    }
}
#[cfg(test)]
mod test_linear_layer {

    use crate::data::Tensor;

    use crate::layer::Layer;
    use crate::layer::LayerRegisterer;

    use std::cell::RefCell;
    use std::rc::Rc;

    use super::LinearLayer;

    #[test]
    fn test_create_layer_find() {
        // 检查nn.Linear 算子是否注册
        let layer_type = "nn.Linear".to_string();
        assert!(LayerRegisterer::check_operator_registration(&layer_type));
    }

    #[test]
    fn test_layer_forward() {
        let input_tensor = Tensor::<f32>::new(&[1, 32]);
        let input_data = vec![Rc::new(RefCell::new(input_tensor))];

        let output_tensor = Tensor::<f32>::new(&[1, 128]);
        let output_data = vec![Rc::new(RefCell::new(output_tensor))];

        let relu_layer = LinearLayer::new(None, 32, 128, true);

        relu_layer
            .forward_with_tensors(&input_data, &output_data)
            .unwrap();
    }
}
