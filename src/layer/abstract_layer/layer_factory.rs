
use lazy_static::lazy_static;
use num_traits::Zero;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Mutex;

use crate::layer::Layer;
use crate::runtime::{RuntimeOperator, SharedRuntimeOperator};

type Creator<A> = fn(SharedRuntimeOperator<A>) -> Rc<dyn Layer<A>>;
type CreateRegistry<A> = HashMap<String, Creator<A>>;
pub struct LayerRegistry<A> {
    pub registry: CreateRegistry<A>,
}
impl<A> LayerRegistry<A>
where
    A: Clone + Zero + PartialOrd + 'static,
{
    fn new() -> Self {
        let registry = HashMap::new();
        let mut layer_registry = LayerRegistry { registry: registry };
        //注册nn.ReLu 算子
        use crate::layer::details::relu::ReLULayer;
        layer_registry.register_creator("nn.ReLu".to_string(), ReLULayer::<A>::get_instance);

        layer_registry
    }
    pub fn register_creator(&mut self, layer_type: String, creator: Creator<A>) {
        self.registry.insert(layer_type, creator);
    }

    pub fn get(&self, layer_type: &String) -> Option<& Creator<A>>{
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
    /// let registry = LayerRegisterer::get_registry();
    ///
    /// ```
    pub fn get_registry() -> &'static Mutex<LayerRegistry<f32>> {
        &OPERATOR_REGISTRY_F32
    }

    pub fn create_layer(operator:& Rc<RefCell<RuntimeOperator<f32>>>) -> Rc<dyn Layer<f32>> {
        let registry= & mut LayerRegisterer::get_registry().lock().unwrap();
        let layer_type = &operator.borrow().type_name;
        let creator = registry.get(layer_type).unwrap();
        let layer_ptr: Rc<dyn Layer<f32>> = creator(operator.clone());
        layer_ptr
    }
}
// impl  {

// }
#[cfg(test)]
mod test_runtime_operator_util {
    use super::*;
    #[test]
    fn test_get_relu() {
        let _relu_layer = LayerRegisterer::get_registry();


    }
}
