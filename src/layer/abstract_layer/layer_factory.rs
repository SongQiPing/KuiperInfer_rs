use std::rc::Rc;
use std::collections::HashMap;
use crate::runtime::RuntimeOperator;
use crate::status_code::ParseParameterAttrStatus;
use crate::layer::Layer;

use lazy_static::lazy_static;
use std::sync::Mutex;
type Creator = fn(&Rc<RuntimeOperator<f32>>, &mut Rc<dyn Layer<f32>>) -> ParseParameterAttrStatus;
type CreateRegistry = HashMap<String, Creator>;

lazy_static! {
    static ref REGISTRY_F32: Mutex<CreateRegistry> = Mutex::new(HashMap::new());
}

pub struct LayerRegisterer{
    
}



impl LayerRegisterer{
    
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
    pub fn get_registry() -> &'static Mutex<CreateRegistry>{
        &REGISTRY_F32
    }
    /// 向注册表中注册算子
    ///
    /// # Examples
    ///
    /// ```

    /// ```
    pub fn register_creator<A>(layer_type:&String, creator:& Creator){
        let registry:& mut CreateRegistry = & mut LayerRegisterer::get_registry().lock().unwrap();

        registry.insert(layer_type.clone(), *creator);
    }

    // pub fn create_layer(operator:& Rc<RefCell<RuntimeOperator<f32>>>){
    //     let registry:& mut CreateRegistry = & mut LayerRegisterer::get_registry().lock().unwrap();
    //     let layer_type = &operator.borrow().type_name;
    //     let creator = registry.get(layer_type).unwrap();
    //     let layer_ptr = Rc::new(RefCell::new(dyn Layer<f32>));

    // }

}   
// impl  {
    
// }
#[cfg(test)]
mod test_runtime_operator_util {
    #[test]
    fn test_init_operator_input() {
        // LayerRegisterer::get_registry();
        // let registerer:&CreateRegistry<f32> = & LayerRegisterer::get_registry().lock().unwrap();
        // let registry_f32 = registerer.get_registry().lock().unwrap();
    }
}
