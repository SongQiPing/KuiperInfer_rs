use num_traits::Zero;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use super::pnnx;
use super::RuntimeAttribute;
use super::RuntimeDataType;
use super::RuntimeOperand;
use super::RuntimeOperator;

use super::RuntimeOperatorUtil;
use super::SharedRuntimeOperand;
use super::SharedRuntimeOperator;

use crate::pnnx::SharedOperand;
pub enum GraphState {
    Complete,
    NeedBuild,
    NeedInit,
}
pub struct RuntimeGraph<A> {
    intput_name: String, //计算输入节点的名称
    output_name: String, //计算图输入节点的名称
    param_path: String,  //计算图的结构文件
    bin_path: String,    //计算图的权重文件

    operators: Vec<SharedRuntimeOperator<A>>,
    pub operators_maps: HashMap<String, SharedRuntimeOperator<A>>,
    topo_operators: Vec<SharedRuntimeOperator<A>>,

    graph: Box<pnnx::Graph>,
    graph_state: GraphState,
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
            graph_state: GraphState::NeedInit,
            topo_operators: Vec::new(),
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
            let runtime_operator: SharedRuntimeOperator<A> =
                Rc::new(RefCell::new(RuntimeOperator::<A>::new()));
            runtime_operator.borrow_mut().name = operator.as_ref().borrow().name.clone();
            runtime_operator.borrow_mut().type_name = operator.as_ref().borrow().type_name.clone();

            // 初始化算子中的input
            let inputs: &Vec<SharedOperand> = &operator.as_ref().borrow().inputs;
            self.init_graph_operators_input(inputs, &runtime_operator);

            //记录输出operand的名称
            let outputs = &operator.as_ref().borrow().outputs;
            self.init_graph_operator_output(outputs, &runtime_operator);

            // 初始化算子中的attribute(权重)
            self.init_graph_attrs(&operator.as_ref().borrow().attrs, &runtime_operator);

            // 初始化算子中的parameter
            self.init_graph_params(&operator.as_ref().borrow().params, &runtime_operator);

            self.operators.push(runtime_operator.clone());
            self.operators_maps.insert(
                runtime_operator.as_ref().borrow().name.clone(),
                runtime_operator.clone(),
            );
        }
        self.graph_state = GraphState::NeedBuild;
    }
    pub fn init_graph_params(
        &self,
        param_map: &HashMap<String, pnnx::Parameter>,
        runtime_operator: &SharedRuntimeOperator<A>,
    ) {
        for (key, val) in param_map.iter() {
            runtime_operator
                .borrow_mut()
                .params
                .insert(key.clone(), Rc::new(RefCell::new(val.clone())));
        }
    }

    pub fn init_graph_attrs(
        &self,
        attribute_map: &HashMap<String, pnnx::graph::Attribute>,
        runtime_operator: &SharedRuntimeOperator<A>,
    ) {
        for (key, val) in attribute_map.iter() {
            match val.type_id {
                1 => {
                    let runtime_attribute = RuntimeAttribute {
                        weight_data: val.data.clone(),
                        dtype: RuntimeDataType::TypeFloat32,
                        shape: val.shape.clone(),
                    };
                    runtime_operator
                        .borrow_mut()
                        .attribute
                        .insert(key.clone(), Rc::new(RefCell::new(runtime_attribute)));
                }
                _ => {
                    panic!("Unknown attribute type");
                }
            }
        }
    }
    pub fn init_graph_operator_output(
        &self,
        outputs: &Vec<SharedOperand>,
        runtime_operator: &SharedRuntimeOperator<A>,
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
        inputs: &Vec<SharedOperand>,
        runtime_operator: &SharedRuntimeOperator<A>,
    ) {
        for input in inputs {
            let producer = &input.as_ref().borrow().producer;
            if let Some(producer) = producer {
                let runtime_operand: SharedRuntimeOperand<A> =
                    Rc::new(RefCell::new(RuntimeOperand::<A>::new()));
                runtime_operand.borrow_mut().name = producer.as_ref().borrow().name.clone();
                runtime_operand.borrow_mut().shapes = input.as_ref().borrow().shape.clone();
            }
            let runtime_operand: SharedRuntimeOperand<A> =
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

    pub fn build(&mut self, input_name: String, output_name: String) {
        if let GraphState::Complete = self.graph_state {
            println!("Model has been built already!");
        }

        if let GraphState::NeedInit = self.graph_state {
            self.init();
        }

        //构建图关系
        for current_operator in &self.operators {
            let output_names = &current_operator.as_ref().borrow().output_names.clone();
            for output_name in output_names {
                if self.operators_maps.contains_key(output_name) {
                    let output_name_key = output_name.clone();
                    let next_operator_ref = self.operators_maps[output_name].clone();

                    current_operator
                        .as_ref()
                        .borrow_mut()
                        .output_operators
                        .insert(output_name_key, next_operator_ref);
                }
            }
        }
        //初始化节点的输入和输出空间
        RuntimeOperatorUtil::init_operator_input(&self.operators);
        RuntimeOperatorUtil::init_operator_output(&self.graph.operators, &self.operators);

        // 构建拓扑顺序
        self.topo_operators.clear();
        for (_, operator) in &self.operators_maps {
            if operator.borrow().type_name == "pnnx.Input".to_string()
                && !operator.borrow().has_forward
            {
                Self::reverse_topo(operator, &mut self.topo_operators);
            }
        }
        self.topo_operators.reverse();
        assert_eq!(
            self.topo_operators.len(),
            self.operators.len(),
            "Build wrong topo queue"
        );
        self.graph_state = GraphState::Complete;
        self.intput_name = input_name;
        self.output_name = output_name;
    }

    pub fn reverse_topo(
        root_op: &SharedRuntimeOperator<A>,
        topo_operators: &mut Vec<SharedRuntimeOperator<A>>,
    ) {
        root_op.borrow_mut().has_forward = true;
        for (_, next_operator) in &root_op.borrow().output_operators {
            if !next_operator.borrow().has_forward {
                Self::reverse_topo(next_operator, topo_operators);
            }
        }
        topo_operators.push(root_op.clone());
    }
    pub fn get_topo_queues(&self) -> &Vec<SharedRuntimeOperator<A>> {
        &self.topo_operators
    }
}

#[cfg(test)]
mod test_runrime_graph {
    use super::*;
    #[test]
    fn test_new_graph() {
        let param_path = "model_file/test_linear.pnnx.param".to_string();
        let bin_path = "model_file/test_linear.pnnx.bin".to_string();
        let _runtime_grpah: RuntimeGraph<f32> = RuntimeGraph::<f32>::new(param_path, bin_path);
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

#[cfg(test)]
mod test_graph_topo {
    use super::*;

    #[test]
    fn test_topo() {
        let param_path = "model_file/resnet18_batch1.param".to_string();
        let bin_path = "model_file/resnet18_batch1.pnnx.bin".to_string();
        let mut runtime_grpah: RuntimeGraph<f32> = RuntimeGraph::<f32>::new(param_path, bin_path);
        runtime_grpah.init();
        runtime_grpah.build("pnnx_input_0".to_string(), "pnnx_output_0".to_string());
        let topo_queues = runtime_grpah.get_topo_queues();

        let mut index = 0;
        for operator in topo_queues {
            println!(
                "Index:{}, Type:{}, Name:{}",
                index,
                &operator.borrow().type_name,
                &operator.borrow().name
            );
            index += 1;
        }
    }
    #[test]
    fn test_build_output_ops() {
        let param_path = "model_file/simple_ops.pnnx.param".to_string();
        let bin_path = "model_file/simple_ops.pnnx.bin".to_string();
        let mut runtime_grpah: RuntimeGraph<f32> = RuntimeGraph::<f32>::new(param_path, bin_path);
        runtime_grpah.init();
        runtime_grpah.build("pnnx_input_0".to_string(), "pnnx_output_0".to_string());
        let topo_queues = runtime_grpah.get_topo_queues();

        let mut index = 0;
        for operator in topo_queues {
            println!(
                "Index:{}, Type:{}, Name:{}",
                index,
                &operator.borrow().type_name,
                &operator.borrow().name
            );
            index += 1;
        }
    }
    #[test]
    fn test_build_output_ops2() {
        let param_path = "model_file/simple_ops.pnnx.param".to_string();
        let bin_path = "model_file/simple_ops.pnnx.bin".to_string();
        let mut runtime_grpah: RuntimeGraph<f32> = RuntimeGraph::<f32>::new(param_path, bin_path);
        runtime_grpah.init();
        runtime_grpah.build("pnnx_input_0".to_string(), "pnnx_output_0".to_string());
        let topo_queues = runtime_grpah.get_topo_queues();

        for operator in topo_queues {
            println!("operator name:{}", operator.borrow().name);
            for (output_operator_name, _) in &operator.borrow().output_operators {
                println!("output: {}", output_operator_name);
            }
            println!("-------------------------");
        }
    }
    #[test]
    fn test_build_status() {
        let param_path = "model_file/simple_ops.pnnx.param".to_string();
        let bin_path = "model_file/simple_ops.pnnx.bin".to_string();
        let mut runtime_grpah: RuntimeGraph<f32> = RuntimeGraph::<f32>::new(param_path, bin_path);
        assert!(matches!(runtime_grpah.graph_state, GraphState::NeedInit));
        runtime_grpah.init();
        assert!(matches!(runtime_grpah.graph_state, GraphState::NeedBuild));
        runtime_grpah.build("pnnx_input_0".to_string(), "pnnx_output_0".to_string());
        assert!(matches!(runtime_grpah.graph_state, GraphState::Complete));
    }
    #[test]
    fn test_build_output_tensors() {
        let param_path = "model_file/simple_ops2.pnnx.param".to_string();
        let bin_path = "model_file/simple_ops2.pnnx.bin".to_string();
        let mut runtime_grpah: RuntimeGraph<f32> = RuntimeGraph::<f32>::new(param_path, bin_path);
        assert!(matches!(runtime_grpah.graph_state, GraphState::NeedInit));
        runtime_grpah.init();
        assert!(matches!(runtime_grpah.graph_state, GraphState::NeedBuild));
        runtime_grpah.build("pnnx_input_0".to_string(), "pnnx_output_0".to_string());
        assert!(matches!(runtime_grpah.graph_state, GraphState::Complete));

        let operators = &runtime_grpah.operators;
        for operator in operators {
            println!("name:{}", operator.borrow().name);

            let operands = &operator.borrow().output_operands;
            match operands {
                Some(operand_list) => {
                    let batch_size = operand_list.borrow().datas.len();
                    println!("batch:{}", batch_size);
                    for tensor_index in 0..batch_size {
                        let data = &operand_list.borrow().datas[tensor_index];
                        println!("shape:{:?}", data.borrow().shapes());
                    }
                }
                None => {}
            }
        }
    }
}
