use ndarray::prelude::*;
use num_traits::Zero;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use super::pnnx;
use super::RuntimeAttribute;
use super::RuntimeDataType;
use super::RuntimeOperand;
use super::RuntimeOperator;
use super::RuntimeParameter;

pub struct RuntimeGraph<A> {
    intput_name: String, //计算输入节点的名称
    output_name: String, //计算图输入节点的名称
    param_path: String,  //计算图的结构文件
    bin_path: String,    //计算图的权重文件

    operators: Vec<Rc<RefCell<RuntimeOperator<A>>>>,
    operators_maps: HashMap<String, Rc<RefCell<RuntimeOperator<A>>>>,

    graph: Box<pnnx::Graph>,
}

impl<A> RuntimeGraph<A>
where
    A: Clone + Zero,
{
    pub fn new(param_path: String, bin_path: String) -> Self {
        RuntimeGraph {
            intput_name: String::new(),
            output_name: String::new(),
            param_path: param_path,
            bin_path: bin_path,
            operators: Vec::new(),
            operators_maps: HashMap::new(),
            graph: Box::new(pnnx::Graph::new()),
        }
    }
    pub fn set_bin_path(&mut self, bin_path: String) {
        self.bin_path = bin_path;
    }

    pub fn set_param_path(&mut self, param_path: String) {
        self.param_path = param_path;
    }

    pub fn get_bin_path(&self) -> &String {
        &self.bin_path
    }
    pub fn get_param_path(&self) -> &String {
        &self.param_path
    }
    pub fn init(&mut self) {
        if self.bin_path.is_empty() || self.param_path.is_empty() {
            panic!("The bin path or param path is empty");
        }
        self.graph = Box::new(pnnx::Graph::new());
        self.graph
            .load(&self.param_path.as_str(), &self.bin_path.as_str());

        let operators = &self.graph.operators;
        if operators.is_empty() {
            panic!("Can not read the layers' define");
        }
        self.operators.clear();
        self.operators_maps.clear();

        for operator in operators {
            let runtime_operator: Rc<RefCell<RuntimeOperator<A>>> =
                Rc::new(RefCell::new(RuntimeOperator::<A>::new()));
            runtime_operator.borrow_mut().name = operator.as_ref().borrow().name.clone();
            runtime_operator.borrow_mut().type_name = operator.as_ref().borrow().type_name.clone();

            // 初始化算子中的input
            let inputs: &Vec<Rc<RefCell<pnnx::Operand>>> = &operator.as_ref().borrow().inputs;
            self.init_graph_operators_input(inputs, &runtime_operator);

            //记录输出operand的名称
            let outputs = &operator.as_ref().borrow().outputs;
            self.init_graph_operator_output(outputs, &runtime_operator);

            // 初始化算子中的attribute(权重)
            self.init_graph_attrs(&operator.as_ref().borrow().attrs, &runtime_operator);

            // 初始化算子中的parameter
            self.init_graph_params(& operator.as_ref().borrow().params, & runtime_operator);

            self.operators.push(runtime_operator.clone());
            self.operators_maps.insert(runtime_operator.as_ref().borrow().name.clone(), runtime_operator.clone());

        }
    }
    pub fn init_graph_params(& self, param_map:& HashMap<String, pnnx::Parameter>, runtime_operator: &Rc<RefCell<RuntimeOperator<A>>>){
        for (key, val) in param_map.iter() {
            runtime_operator.borrow_mut().params.insert(key.clone(), Rc::new(RefCell::new(val.clone())));
        }
    }

    pub fn init_graph_attrs(
        &self,
        attribute_map: &HashMap<String, pnnx::graph::Attribute>,
        runtime_operator: &Rc<RefCell<RuntimeOperator<A>>>,
    ) {
        for (key, val) in attribute_map.iter() {
            match val.type_id {
                1  => {
                    let runtime_attribute = RuntimeAttribute {
                        weight_data: val.data.clone(),
                        dtype: RuntimeDataType::TypeFloat32,
                        shape: val.shape.clone(),
                    };
                    runtime_operator.borrow_mut().attribute.insert(key.clone(), Rc::new(RefCell::new(runtime_attribute)));
                    
                }
                _ => {
                    panic!("Unknown attribute type");

                }
            }
        }
    }
    pub fn init_graph_operator_output(
        &self,
        outputs: &Vec<Rc<RefCell<pnnx::Operand>>>,
        runtime_operator: &Rc<RefCell<RuntimeOperator<A>>>,
    ) {
        if outputs.is_empty() {
            return;
        }
        for output_operand in outputs {
            let consumers = &output_operand.as_ref().borrow_mut().consumers;
            for consumer in consumers {
                runtime_operator
                    .as_ref()
                    .borrow_mut()
                    .output_names
                    .push(consumer.as_ref().borrow().name.clone());
            }
        }
    }

    pub fn init_graph_operators_input(
        &self,
        inputs: &Vec<Rc<RefCell<pnnx::Operand>>>,
        runtime_operator: &Rc<RefCell<RuntimeOperator<A>>>,
    ) {
        for input in inputs {
            let producer = &input.as_ref().borrow().producer;
            if let Some(producer) = producer {
                let runtime_operand: Rc<RefCell<RuntimeOperand<A>>> =
                    Rc::new(RefCell::new(RuntimeOperand::<A>::new()));
                runtime_operand.borrow_mut().name = producer.as_ref().borrow().name.clone();
                runtime_operand.borrow_mut().shapes = input.as_ref().borrow().shape.clone();
            }
            let runtime_operand: Rc<RefCell<RuntimeOperand<A>>> =
                Rc::new(RefCell::new(RuntimeOperand::<A>::new()));
            runtime_operand.borrow_mut().name =
                producer.as_ref().expect("REASON").borrow().name.clone();
            runtime_operand.borrow_mut().shapes = input.as_ref().borrow().shape.clone();

            match input.as_ref().borrow().type_id {
                1 => {
                    runtime_operand.borrow_mut().data_type = RuntimeDataType::TypeFloat32;
                }
                0 => {
                    runtime_operand.borrow_mut().data_type = RuntimeDataType::TypeUnknown;
                }
                _ => {
                    panic!("Unknown input operand type: ");
                }
            }

            runtime_operator.borrow_mut().input_operands.insert(
                producer.as_ref().unwrap().borrow().name.clone(),
                runtime_operand.clone(),
            );
            runtime_operator
                .borrow_mut()
                .input_operands_seq
                .push(runtime_operand.clone());
        }
    }
}

#[cfg(test)]
mod test_runrime_graph {
    use super::*;
    #[test]
    fn test_new_graph() {
        let param_path = "model_file/test_linear.pnnx.param".to_string();
        let bin_path = "model_file/test_linear.pnnx.bin".to_string();
        let runtime_grpah: RuntimeGraph<f32> = RuntimeGraph::<f32>::new(param_path, bin_path);
    }
    #[test]
    #[should_panic(expected = "The bin path or param path is empty")]
    fn test_init_para_path_empty() {
        let param_path = String::new();
        let bin_path = "model_file/test_linear.pnnx.bin".to_string();
        let mut runtime_grpah: RuntimeGraph<f32> = RuntimeGraph::<f32>::new(param_path, bin_path);
        runtime_grpah.init();
    }
    #[test]
    #[should_panic(expected = "The bin path or param path is empty")]
    fn test_init_bin_path_empty() {
        let param_path = "model_file/test_linear.pnnx.param".to_string();
        let bin_path = String::new();
        let mut runtime_grpah: RuntimeGraph<f32> = RuntimeGraph::<f32>::new(param_path, bin_path);
        runtime_grpah.init();
    }
    #[test]
    fn test_init() {
        let param_path = "model_file/test_linear.pnnx.param".to_string();
        let bin_path = "model_file/test_linear.pnnx.bin".to_string();
        let mut runtime_grpah: RuntimeGraph<f32> = RuntimeGraph::<f32>::new(param_path, bin_path);
        runtime_grpah.init();
    }

}
