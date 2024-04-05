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
use crate::Tensor;
use log::error;
use log::info;
use ndarray::prelude::*;
use ndarray::LinalgScalar;
use num_traits::Bounded;
use num_traits::Zero;

pub struct ConvolutionLayer<A>
where
    A: Clone + Zero,
{
    runtime_operator: RuntimeOperatorData<A>,
    layer_name: String,
    in_channel: usize,
    out_channel: usize,
    padding_h: usize,
    padding_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    groups: usize,
    use_bias: bool,
    weights: WeightData<A>,
    bias: Option<WeightData<A>>,
    kernel_matrix: SharedTensor<A>,
}
impl<A> ConvolutionLayer<A>
where
    A: Clone + LinalgScalar + PartialOrd + std::ops::Neg + Bounded + std::fmt::Debug,

    f32: From<A>,
    A: From<f32>,
{
    /// im2col
    /// 将图像特征图转换为矩阵的形式
    ///
    /// # Arguments
    /// * `input` - 输入的特征图像
    /// * `kernel_h` - 卷积核的高度
    /// * `kernel_w` - 卷积核的宽度
    /// * `input_h` - 输入特征图的高度
    /// * `input_w` - 输入特征图的宽度
    /// * `padding_h` - 卷积的填充
    /// * `padding_w` - 卷积的填充
    /// * `input_c_group` - 每个group处理的通道数量
    /// * `group` - 当前进⾏Im2Col的组数(group)
    /// * `row_len` - 卷积核的计算次数
    /// * `col_len` - 卷积核的计算次数
    /// ```
    /// ```
    fn im2col(
        inputs: SharedTensor<A>,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        input_h: usize,
        input_w: usize,
        padding_h: usize,
        padding_w: usize,
        input_c_group: usize,
        group_index: usize,
        row_len: usize,
        _col_len: usize,
    ) -> SharedTensor<A> {
        let mut input_matrix_ndarry =
            ArrayD::<A>::zeros(IxDyn(&[row_len, input_c_group, kernel_h, kernel_w]));
        info!("{:?}", input_matrix_ndarry.shape());
        let input_padding_h: usize = input_h + 2 * padding_h;
        let input_padding_w = input_w + 2 * padding_w;
        let paddding_val: A = Zero::zero();
        for ic in 0..input_c_group {
            let mut current_row = 0;
            let cur_channel = ic + group_index * input_c_group;
            for row in (0..=input_padding_h - kernel_h).step_by(stride_h) {
                for col in (0..=input_padding_w - kernel_w).step_by(stride_w) {
                    for r in 0..kernel_h {
                        for c in 0..kernel_w {
                            let cur_row = row + r;
                            let cur_col = col + c;

                            let cur_val = if cur_row >= padding_h
                                && cur_col >= padding_w
                                && cur_row < input_h + padding_h
                                && cur_col < input_w + padding_w
                            {
                                inputs
                                    .as_ref()
                                    .borrow()
                                    .index(&[cur_channel, cur_row - padding_h, cur_col - padding_w])
                                    .clone()
                            } else {
                                paddding_val.clone()
                            };
                            let matrix_offset = IxDyn(&[current_row, ic, r, c]);
                            input_matrix_ndarry[matrix_offset] = cur_val;
                        }
                    }
                    current_row = current_row + 1;
                }
            }
        }
        let shape = IxDyn(&[row_len, input_c_group * kernel_h * kernel_w]);
        let input_matrix = input_matrix_ndarry.into_shape(shape).unwrap();
        let input_matrix = input_matrix.t().to_owned();

        Tensor::<A>::from_ndarry(input_matrix).shared_self()
    }
    ///  执行带有偏置的卷积运算
    ///
    /// # Arguments
    /// * `input_matrix` 图像或特征图的卷积矩阵
    /// * `kernel_matrix` 卷积矩阵
    /// * `bias` 偏置矩阵
    fn conv_gemm_bias(
        input_matrix: SharedTensor<A>,
        kernel_matrix: &SharedTensor<A>,
        bias: &Option<WeightData<A>>,
    ) -> ArrayD<A> {
        let input_matrix = input_matrix.as_ref().borrow().data().clone();
        let input_matrix: ArrayBase<ndarray::OwnedRepr<A>, Dim<[usize; 2]>> =
            input_matrix.clone().into_dimensionality().unwrap();
        let kernel_matrix = kernel_matrix.as_ref().borrow().data().clone();
        let kernel_matrix: ArrayBase<ndarray::OwnedRepr<A>, Dim<[usize; 2]>> =
            kernel_matrix.clone().into_dimensionality().unwrap();

        let mut output_tensor: ndarray::ArrayBase<ndarray::OwnedRepr<A>, ndarray::Dim<[usize; 2]>> =
            kernel_matrix.dot(&input_matrix);

        if let Some(bias_data) = bias {
            let bias_data = bias_data.get().borrow().data().clone();
            let out_channel = kernel_matrix.shape()[0];

            let bias_data = bias_data.into_shape(IxDyn(&[out_channel, 1])).unwrap();
            let bias_data: ArrayBase<ndarray::OwnedRepr<A>, Dim<[usize; 2]>> =
                bias_data.clone().into_dimensionality().unwrap();

            output_tensor = output_tensor + bias_data;
        }
        let output_tensor = output_tensor.into_dyn();
        output_tensor
    }
}
impl<A> ConvolutionLayer<A>
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
        output_channel: usize,
        in_channel: usize,
        kernel_h: usize,
        kernel_w: usize,
        padding_h: usize,
        padding_w: usize,
        stride_h: usize,
        stride_w: usize,
        groups: usize,
        use_bias: bool,
    ) -> Self {
        let runtime_operator = match runtime_operator {
            Some(runtime_operator) => runtime_operator,
            None => RuntimeOperatorData::new(),
        };

        let weight_data =
            WeightData::<A>::init_param(output_channel, in_channel / groups, kernel_h, kernel_w);

        let bias_data = match use_bias {
            true => Some(WeightData::<A>::init_param(output_channel, 1, 1, 1)),
            false => None,
        };
        let kernel_matrix = Tensor::<A>::new(&[output_channel, in_channel * kernel_h * kernel_w]);

        let conv2d = ConvolutionLayer {
            runtime_operator: runtime_operator,
            layer_name: "nn.Conv2d".to_string(),
            in_channel: in_channel,
            out_channel: output_channel,
            padding_h: padding_h,
            padding_w: padding_w,
            kernel_h: kernel_h,
            kernel_w: kernel_w,
            stride_h: stride_h,
            stride_w: stride_w,
            groups: groups,
            use_bias: use_bias,
            weights: weight_data,
            bias: bias_data,
            kernel_matrix: Rc::new(RefCell::new(kernel_matrix)),
        };

        conv2d
    }

    pub fn load_weight(
        &mut self,
        attribute_map: &HashMap<String, Rc<RefCell<RuntimeAttribute>>>,
    ) -> Result<(), LayerError> {
        match attribute_map.get(&"weight".to_string()) {
            Some(weight_attibute) => {
                let vec_data: Vec<A> = weight_attibute.as_ref().borrow_mut().get::<A>(false);
                let shape: [usize; 4] = [
                    self.out_channel,
                    self.in_channel,
                    self.kernel_h,
                    self.kernel_h,
                ];
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
                let shape: [usize; 4] = [self.out_channel, 1, 1, 1];
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
        self.init_im2col_weight()?;
        // //加载偏置

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

        let in_channel = ParameterData::get_in_channels(params_map)?;
        let out_channel = ParameterData::get_out_channels(params_map)?;
        //获取stride的参数
        let stride = ParameterData::get_stride(params_map)?;
        // 获取padding的参数
        let padding = ParameterData::get_padding(params_map)?;
        // 获取kernel_size
        let kernel_size = ParameterData::get_kernel_size(params_map)?;

        let _dilation = ParameterData::get_dilation(params_map)?;
        let _padding_mode = ParameterData::get_padding_mode(params_map)?;
        let groups = ParameterData::get_groups(params_map)?;
        // 获取bias
        let has_bias = ParameterData::get_has_bias(params_map)?;
        let mut conv_2d_layer = ConvolutionLayer::<A>::new(
            Some(RuntimeOperatorData::new_from(runtime_operator.clone())),
            out_channel.try_into().unwrap(),
            in_channel.try_into().unwrap(),
            kernel_size[0].try_into().unwrap(),
            kernel_size[1].try_into().unwrap(),
            padding[0].try_into().unwrap(),
            padding[1].try_into().unwrap(),
            stride[0].try_into().unwrap(),
            stride[1].try_into().unwrap(),
            groups.try_into().unwrap(),
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
        info!(
            "InChannel: ({}), OutChannel:({})",
            self.in_channel, self.out_channel
        );
        info!("Padding (H, W): ({}, {})", self.padding_h, self.padding_w);
        info!("k Size (H, W): ({}, {})", self.kernel_h, self.kernel_w);
        info!("Stride (H, W): ({}, {})", self.stride_h, self.stride_w);
        info!("Groups (): ({})", self.groups);
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
        let input_c_group = input_c / self.groups;

        // // 检验  输出的形状是否符合要求
        let row_len = output_h * output_w;
        let col_len = input_c_group * self.kernel_h * self.stride_w;

        self.check_output_data_shape(outputs, output_h, output_w)?;
        for group_index in 0..self.groups {
            let input_matrix = Self::im2col(
                inputs.clone(),
                self.kernel_h,
                self.kernel_w,
                self.stride_h,
                self.stride_w,
                input_h,
                input_w,
                self.padding_h,
                self.padding_w,
                input_c_group,
                group_index,
                row_len,
                col_len,
            );
            let out_data_ndarry = Self::conv_gemm_bias(
                input_matrix.clone(),
                &self.kernel_matrix.clone(),
                &self.bias,
            );
            let output_shape = IxDyn(&[self.out_channel, output_h, output_w]);
            let out_data_ndarry = out_data_ndarry.into_shape(output_shape).unwrap();
            outputs.as_ref().borrow_mut().set_data(out_data_ndarry);
        }

        Ok(())
    }
    // /// 计算运算之后的形状
    fn calc_output_shape(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let padding_h = self.padding_h;
        let kernel_h = self.kernel_h;
        let stride_h = self.stride_h;
        let input_padded_h = input_h + 2 * padding_h;

        let output_h: usize = (input_padded_h - kernel_h) / stride_h + 1;

        let padding_w = self.padding_w;
        let kernel_w = self.kernel_w;
        let stride_w = self.stride_w;
        let input_padded_w = input_w + 2 * padding_w;
        let output_w: usize = (input_padded_w - kernel_w) / stride_w + 1;
        (output_h, output_w)
    }
    fn check_output_data_shape(
        &self,
        outputs: &SharedTensor<A>,
        output_h: usize,
        output_w: usize,
    ) -> Result<(), LayerError> {
        if outputs.borrow().rows() != output_h || outputs.borrow().cols() != output_w {
            error!(
                "now output data shape is {:?}, expect outdata shape is{:?} ",
                (outputs.borrow().rows(), outputs.borrow().cols()),
                (output_h, output_w)
            );
            return Err(LayerError::InferFailedOutputSizeError);
        }
        Ok(())
    }
    pub fn init_im2col_weight(&mut self) -> Result<(), LayerError> {
        let kernel_count = self.out_channel;
        let kernel_c = self.in_channel;
        let kernel_h = self.kernel_h;
        let kernel_w = self.kernel_w;
        let row_len = kernel_h * kernel_w;
        let kernel_matrix_shape = IxDyn(&[kernel_count, row_len * kernel_c]);

        self.kernel_matrix = if self.groups == 1 {
            let kernel_ndarry = self.weights.get().borrow().data().clone();

            let kernel_ndarry = kernel_ndarry.into_shape(kernel_matrix_shape).unwrap();

            Tensor::from_ndarry(kernel_ndarry).shared_self()
        } else {
            // group != 1
            let kernel_count_group = kernel_count / self.groups;
            let part_kernel_matrix_shape = IxDyn(&[kernel_count_group, row_len * kernel_c]);
            // 获取数据
            let kernel_ndarry = self.weights.get().borrow().data().clone();
            let kernel_ndarry = kernel_ndarry.into_shape(part_kernel_matrix_shape).unwrap();
            // 重复维度
            let shape = IxDyn(&[self.groups, kernel_count_group, row_len * kernel_c]);
            let kernel_ndarry = kernel_ndarry.broadcast(shape).unwrap().to_owned();
            // 转换成形状为 [kernel_count, row_len * kernel_c]
            let kernel_ndarry = kernel_ndarry.into_shape(kernel_matrix_shape).unwrap();
            Tensor::from_ndarry(kernel_ndarry).shared_self()
        };

        Ok(())
    }
}
use std::convert::From;

impl<A> Layer<A> for ConvolutionLayer<A>
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
            self.batch_forward_wight_tensors(input_data, output_data)?;
        }
        Ok(())
    }

    fn layer_name(&self) -> &String {
        &self.layer_name
    }
}
#[cfg(test)]
mod test_conv2d_layer {

    use log::info;
    use ndarray::Array1;
    use ndarray::Array3;
    use ndarray::Array4;
    use ndarray::ArrayD;
    use ndarray::Dim;
    use ndarray::IxDyn;
    use num_traits::Zero;

    use super::ConvolutionLayer;
    use crate::data::Tensor;
    use crate::layer::abstract_layer::layer::RuntimeOperatorGetterSetter;
    use crate::layer::Layer;
    use crate::layer::LayerRegisterer;
    use crate::runtime::RuntimeGraph;
    use crate::runtime::RuntimeOperand;
    use crate::runtime::RuntimeOperator;
    use crate::runtime::SharedRuntimeOperator;
    use ndarray::prelude::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn conv2d_ndarray(
        input: Array3<f32>,
        weight: Array4<f32>,
        bias: Array1<f32>,
        padding_h: usize,
        padding_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Array3<f32> {
        let (_, input_h, input_w) = input.dim();
        let (out_channel, in_channel, kernel_h, kernel_w) = weight.dim();
        let output_h = (input_h + 2 * padding_h - kernel_h) / stride_h + 1;
        let output_w = (input_w + 2 * padding_w - kernel_w) / stride_w + 1;
        let input_padding_h = input_h + 2 * padding_h;
        let input_padding_w = input_w + 2 * padding_w;

        let mut output = Array3::<f32>::zeros((out_channel, output_h, output_w));

        for i in 0..out_channel {
            for h in 0..output_h {
                for w in 0..output_w {
                    let mut value = bias[i];
                    for d in 0..in_channel {
                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let input_padding_h_idx = h * stride_h + kh;
                                let input_padding_w_idx = w * stride_w + kw;

                                if input_padding_h_idx >= padding_h
                                    && input_padding_h_idx < input_padding_h - padding_h
                                    && input_padding_w_idx >= padding_w
                                    && input_padding_w_idx < input_padding_w - padding_w
                                {
                                    let input_h_idx = input_padding_h_idx - padding_h;
                                    let input_w_idx = input_padding_w_idx - padding_w;
                                    value += input[[d, input_h_idx, input_w_idx]]
                                        * weight[[i, d, kh, kw]];
                                }
                            }
                        }
                    }
                    output[[i, h, w]] = value;
                }
            }
        }

        output
    }
    #[test_log::test]
    fn test_im2col() {
        use ndarray::prelude::*;
        let brr = ArrayD::from_shape_fn(vec![2, 4, 4], |d| {
            (d[0] as f32) * 16.0 + (d[1] as f32) * 4.0 + d[2] as f32
        });
        println!("{:?}", &brr);
        let input_tensor = Tensor::from_ndarry(brr).shared_self();

        let kernel_h = 3;
        let kernel_w = 3;
        let stride_h = 1;
        let stride_w = 1;
        let input_h = 4;
        let input_w = 4;
        let padding_h = 1;
        let padding_w = 1;
        let input_c_group = 2;
        let group_index = 0;
        let row_len = 16;
        let col_len = 4;

        let out_tensor = ConvolutionLayer::im2col(
            input_tensor,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            input_h,
            input_w,
            padding_h,
            padding_w,
            input_c_group,
            group_index,
            row_len,
            col_len,
        );

        println!("{:?}", out_tensor.as_ref().borrow().data.shape());
        println!("{:?}", out_tensor.as_ref().borrow().data);
    }
    // 转换函数
    unsafe fn static_cast<A>(base: &dyn Layer<A>) -> &ConvolutionLayer<A>
    where
        A: Clone + Zero,
    {
        &*(base as *const dyn Layer<A> as *const ConvolutionLayer<A>)
    }
    fn assert_ndarrays_approx_equal(arr1: ArrayView3<f32>, arr2: ArrayView3<f32>, epsilon: f32) {
        // 检查数组形状是否相同
        assert_eq!(arr1.shape(), arr2.shape());

        // 遍历数组并逐个比较元素
        for (&elem1, &elem2) in arr1.iter().zip(arr2.iter()) {
            // 检查元素之间的差异是否在指定的精度范围内
            assert!(
                (elem1 - elem2).abs() <= epsilon,
                "Arrays are not approximately equal"
            );
        }
    }
    #[test_log::test]
    fn test_conv2d_forward_groups1() {
        let runtime_operator = get_test_conv2d_operator_from_pnnx();
        let mut _layer = ConvolutionLayer::get_instance(runtime_operator.clone()).unwrap();

        _layer.forward().unwrap();
        let conv_2d_layer = unsafe {
            let layer = &(*_layer);

            let conv_2d_layer: &ConvolutionLayer<f32> = static_cast(layer);
            conv_2d_layer
        };
        // input_ndarry
        let input_ndarry = conv_2d_layer.runtime_operator.prepare_input_tensor()[0]
            .as_ref()
            .borrow()
            .data()
            .clone();
        let input_ndarry: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, Dim<[usize; 3]>> =
            input_ndarry.clone().into_dimensionality().unwrap();
        // 得到kernel_ndarry
        let kernel_ndarry = conv_2d_layer.weights.weights.clone();
        let kernel_ndarry: ndarray::ArrayBase<
            ndarray::OwnedRepr<f32>,
            ndarray::Dim<ndarray::IxDynImpl>,
        > = kernel_ndarry.as_ref().borrow().data.clone();
        let kernel_ndarry: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, Dim<[usize; 4]>> =
            kernel_ndarry.clone().into_dimensionality().unwrap();

        // 获得bias
        let bias = match &conv_2d_layer.bias {
            Some(bias) => bias.weights.clone(),
            None => {
                panic!()
            }
        };
        let bias_ndarry: ndarray::ArrayBase<
            ndarray::OwnedRepr<f32>,
            ndarray::Dim<ndarray::IxDynImpl>,
        > = bias.as_ref().borrow().data.clone();
        let shape = IxDyn(&[bias_ndarry.shape()[0]]);
        let bias_ndarry = bias_ndarry.into_shape(shape).unwrap();
        let bias_ndarry: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, Dim<[usize; 1]>> =
            bias_ndarry.clone().into_dimensionality().unwrap();

        // Set padding and stride
        let padding_h = 1;
        let padding_w = 1;
        let stride_h = 1;
        let stride_w = 1;

        // Perform convolution
        let expected_output = conv2d_ndarray(
            input_ndarry.clone(),
            kernel_ndarry,
            bias_ndarry,
            padding_h,
            padding_w,
            stride_h,
            stride_w,
        );

        let output_ndarry = conv_2d_layer.runtime_operator.prepare_output_tensor()[0]
            .as_ref()
            .borrow()
            .data()
            .clone();
        let output_ndarry: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, Dim<[usize; 3]>> =
            output_ndarry.clone().into_dimensionality().unwrap();
        info!("input_ndarry:{:?}", &input_ndarry[[0, 0, 0]]);
        info!("output_ndarrry shape:{:?}", output_ndarry.shape());
        info!("expected_output shape:{:?}", expected_output.shape());
        assert_ndarrays_approx_equal(output_ndarry.view(), expected_output.view(), 1e-6);
        // assert_eq!(output_ndarry, expected_output);
    }

    fn get_test_conv2d_operator_from_pnnx() -> SharedRuntimeOperator<f32> {
        let param_path = "model_file/simple_ops2.pnnx.param".to_string();
        let bin_path = "model_file/simple_ops2.pnnx.bin".to_string();
        let mut runtime_grpah: RuntimeGraph = RuntimeGraph::new(param_path, bin_path);

        runtime_grpah.init();

        runtime_grpah.build("pnnx_input_0".to_string(), "pnnx_output_0".to_string());

        let conv_2d_layer = runtime_grpah
            .operators_maps
            .get(&"op1".to_string())
            .unwrap();
        let mut operand_operand = RuntimeOperand::<f32>::new();

        for _ in 0..2 {
            // let tensor = ArrayD::random(IxDyn(&[3, 16, 16]), Uniform::<f32>::new(1.0., 1.0));
            let tensor = ArrayD::ones(IxDyn(&[3, 16, 16]));
            let tensor = Tensor::from_ndarry(tensor).shared_self();
            operand_operand.datas.push(tensor);
        }
        let operand_operand = Rc::new(RefCell::new(operand_operand));
        conv_2d_layer
            .as_ref()
            .borrow_mut()
            .input_operands_seq
            .push(operand_operand);
        conv_2d_layer.clone()
    }

    #[test_log::test]
    fn test_create_layer() {
        let runtime_operator: Rc<RefCell<RuntimeOperator<f32>>> =
            get_test_conv2d_operator_from_pnnx();
        let _layer = ConvolutionLayer::get_instance(runtime_operator.clone());
    }
    #[test]
    fn test_create_layer_find() {
        // 检查nn.Conv2d 算子是否注册
        let layer_type = "nn.Conv2d".to_string();
        assert!(LayerRegisterer::check_operator_registration(&layer_type));
    }
    #[test_log::test]
    fn test_create_layer_conv2d_forward_from_pnnx() {
        let runtime_operator = get_test_conv2d_operator_from_pnnx();
        let mut _layer = LayerRegisterer::create_layer(&runtime_operator);

        _layer.forward().unwrap();
    }
}
