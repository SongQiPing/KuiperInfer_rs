use crate::layer::Layer;
use crate::runtime::{RuntimeOperator, SharedRuntimeOperator};
use lazy_static::lazy_static;
use ndarray::LinalgScalar;
use num_traits::Bounded;
use std::cell::RefCell;
use std::collections::HashMap;

use std::rc::Rc;
use std::sync::Mutex;

use crate::layer::LayerError;

type Creator<A> = fn(SharedRuntimeOperator<A>) -> Result<Rc<dyn Layer<A>>, LayerError>;
type CreateRegistry<A> = HashMap<String, Creator<A>>;
pub struct LayerRegistry<A> {
    pub registry: CreateRegistry<A>,
}
pub trait LayerInstance {
    fn get_instance<A>(
        &self,
        runtime_operator: SharedRuntimeOperator<A>,
    ) -> Result<Rc<dyn Layer<A>>, LayerError>;
}

pub trait LayerCreate {
    fn get_instance<A>(
        &self,
        runtime_operator: SharedRuntimeOperator<A>,
    ) -> Result<Rc<dyn Layer<A>>, LayerError>
    where
        Self: Sized;
}

impl<A> LayerRegistry<A>
where
    A: Clone + LinalgScalar + PartialOrd + Bounded + std::ops::Neg + std::fmt::Debug,
    f32: From<A>,
    A: From<f32>,
{
    fn new() -> Self {
        let registry = HashMap::new();
        let mut layer_registry = LayerRegistry { registry };
        //注册nn.ReLu 算子
        use crate::layer::details::relu::ReLULayer;
        layer_registry.register_creator("nn.ReLU".to_string(), ReLULayer::<A>::get_instance);
        use crate::layer::details::sigmoid::SigmoidLayer;
        layer_registry.register_creator("nn.Sigmoid".to_string(), SigmoidLayer::<A>::get_instance);
        use crate::layer::details::maxpooling::MaxPoolingLayer;
        layer_registry.register_creator(
            "nn.MaxPool2d".to_string(),
            MaxPoolingLayer::<A>::get_instance,
        );
        use crate::layer::details::convolution::ConvolutionLayer;
        layer_registry
            .register_creator("nn.Conv2d".to_string(), ConvolutionLayer::<A>::get_instance);
        use crate::layer::details::expression::ExpressionLayer;
        layer_registry.register_creator(
            "pnnx.Expression".to_string(),
            ExpressionLayer::<A>::get_instance,
        );
        use crate::layer::details::linear::LinearLayer;
        layer_registry.register_creator("nn.Linear".to_string(), LinearLayer::<A>::get_instance);
        layer_registry
    }
    pub fn register_creator(&mut self, layer_type: String, creator: Creator<A>) {
        self.registry.insert(layer_type, creator);
    }

    pub fn get(&self, layer_type: &String) -> Option<&Creator<A>> {
        self.registry.get(layer_type)
    }
}
lazy_static! {
    static ref OPERATOR_REGISTRY_F32: Mutex<LayerRegistry<f32>> =
        Mutex::new(LayerRegistry::<f32>::new());
}

pub struct LayerRegisterer {}

impl LayerRegisterer {
    ///  返回算子的注册表
    /// # Return
    ///   算子的注册表
    ///
    /// # Examples
    ///
    /// ```
    /// use kuiper_infer::layer::LayerRegisterer;
    /// use kuiper_infer::layer::details::relu::ReLULayer;
    /// LayerRegisterer::register_creator("nn.ReLU".to_string(), ReLULayer::<f32>::get_instance);
    /// ```
    pub fn get_registry() -> &'static Mutex<LayerRegistry<f32>> {
        &OPERATOR_REGISTRY_F32
    }
    ///  注册算子
    /// # Examples
    ///
    /// ```
    /// use kuiper_infer::layer::LayerRegisterer;
    /// use kuiper_infer::layer::details::relu::ReLULayer;
    /// LayerRegisterer::register_creator("nn.ReLU".to_string(), ReLULayer::<f32>::get_instance);
    /// ```
    pub fn register_creator(layer_type: String, creator: Creator<f32>) {
        OPERATOR_REGISTRY_F32
            .lock()
            .unwrap()
            .register_creator(layer_type, creator);
    }
    ///  从注册器中取出算子
    ///
    /// # Examples
    ///
    /// ```
    /// use kuiper_infer::layer::LayerRegisterer;
    /// use kuiper_infer::runtime::RuntimeOperator;
    /// use std::rc::Rc;
    /// use std::cell::RefCell;
    /// let mut runtime_operator = RuntimeOperator::<f32>::new();
    /// runtime_operator.type_name = "nn.ReLU".to_string();
    /// let runtime_operator = Rc::new(RefCell::new(runtime_operator));
    /// let _relu_layer = LayerRegisterer::create_layer(&runtime_operator);
    /// ```
    pub fn create_layer(operator: &Rc<RefCell<RuntimeOperator<f32>>>) -> Rc<dyn Layer<f32>> {
        let registry = &mut LayerRegisterer::get_registry().lock().unwrap();
        let layer_type = &operator.borrow().type_name;
        let creator = match registry.get(layer_type) {
            Some(creator) => creator.clone(),
            None => {
                panic!("{} is not register", layer_type);
            }
        };
        let layer_ptr: Rc<dyn Layer<f32>> = creator(operator.clone()).unwrap();

        layer_ptr
    }
    /// 检查算子是否在注册注册表中
    ///
    /// # Examples
    ///
    /// ```
    /// use kuiper_infer::layer::LayerRegisterer;
    /// let layer_type = "nn.ReLU".to_string();
    /// assert!(LayerRegisterer::check_operator_registration(&layer_type));
    /// ```
    pub fn check_operator_registration(layer_type: &String) -> bool {
        let registry = &mut LayerRegisterer::get_registry().lock().unwrap();
        registry.get(layer_type).is_some()
    }
}

#[cfg(test)]
mod test_runtime_operator_util {
    use super::*;
    #[test]
    fn test_get_relu() {
        let _relu_layer = LayerRegisterer::get_registry();
        use crate::layer::LayerRegisterer;
        use crate::runtime::RuntimeOperator;
        let mut runtime_operator = RuntimeOperator::<f32>::new();
        runtime_operator.type_name = "nn.ReLU".to_string();
        let runtime_operator = Rc::new(RefCell::new(runtime_operator));
        let _relu_layer: Rc<dyn Layer<f32>> = LayerRegisterer::create_layer(&runtime_operator);
    }
}
