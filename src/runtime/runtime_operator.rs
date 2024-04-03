use crate::data::torch_utils::create_tensor;
use crate::data::torch_utils::create_tensor_1;
use crate::data::torch_utils::create_tensor_2;
use crate::layer::Layer;
use crate::pnnx::Parameter;
use crate::pnnx::SharedOperator;

use super::RuntimeAttribute;
use super::RuntimeDataType;
use super::RuntimeOperand;
use super::SharedRuntimeOperand;
use crate::pnnx::SharedOperand;
use num_traits::Zero;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
fn check_shape(shapes: &Vec<usize>) {
    assert!(
        shapes.len() == 2 || shapes.len() == 3 || shapes.len() == 4,
        "Unsupported tensor shape sizes: "
    );
}

pub type SharedRuntimeOperator<A> = Rc<RefCell<RuntimeOperator<A>>>;

pub struct RuntimeOperator<A> {
    pub has_forward: bool,
    pub name: String,                    // 计算节点的名称
    pub type_name: String,               // 计算节点的类型
    pub layer: Option<Rc<dyn Layer<A>>>, // 节点对应的计算Layer

    pub output_names: Vec<String>, // 节点的输出节点名称
    pub output_operands: Option<SharedRuntimeOperand<A>>, // 节点的输出操作数

    pub input_operands: HashMap<String, SharedRuntimeOperand<A>>, // 节点的输入操作数
    pub input_operands_seq: Vec<SharedRuntimeOperand<A>>,         // 节点的输入操作数，顺序排列
    pub output_operators: HashMap<String, SharedRuntimeOperator<A>>, // 输出节点的名字和节点对应

    pub params: HashMap<String, Rc<RefCell<Parameter>>>, // 算子的参数信息
    pub attribute: HashMap<String, Rc<RefCell<RuntimeAttribute>>>, // 算子的属性信息，内含权重信息
}

impl<A> RuntimeOperator<A> {
    pub fn new() -> Self {
        RuntimeOperator {
            has_forward: false,
            name: String::new(),
            type_name: String::new(),
            layer: None,

            output_names: Vec::new(),
            output_operands: None,
            input_operands: HashMap::new(),
            input_operands_seq: Vec::new(),
            output_operators: HashMap::new(),
            params: HashMap::new(),
            attribute: HashMap::new(),
        }
    }
}

pub struct RuntimeOperatorUtil {}
impl RuntimeOperatorUtil {
    /**
     * 如果图是第一次运行，则根据节点输入operand的形状准备好后续Layer计算中所需要的Tensor
     * 如果图是第二次以上运行，则检查输入operand的形状和operand中张量的形状是否匹配
     * @param operators 计算图中的计算节点
     */
    pub fn init_operator_input<A>(operators: &Vec<SharedRuntimeOperator<A>>) {
        for operator in operators {
            let input_operand_map = &operator.as_ref().borrow().input_operands;
            for (_operand_name, operand) in input_operand_map {
                let operand_type = &operand.borrow().data_type.clone();
                match operand_type {
                    RuntimeDataType::TypeFloat32 => {}
                    _ => {
                        panic!("The graph only support float32 yet!{:?}", operand_type);
                    }
                }
                let input_operand_shape = &operand.borrow().shapes.clone();
                //得到初始化的空间
                let _input_datas = &operand.borrow_mut().datas;
                // println!("input_data:{}", input_datas.len());
                //检查形状是否符合要求
                let _batch = input_operand_shape[0];
                check_shape(&input_operand_shape);

                // TODO:
            }
        }
    }
    pub fn init_operator_output<A: Zero + Clone>(
        pnnx_operators: &Vec<SharedOperator>,
        operators: &Vec<SharedRuntimeOperator<A>>,
    ) {
        assert!(!pnnx_operators.is_empty() && !operators.is_empty());
        assert!(pnnx_operators.len() == operators.len());
        //检查是否是唯一的输出值
        for i in 0..pnnx_operators.len() {
            let operands = &pnnx_operators[i].as_ref().borrow().outputs;
            assert!(operands.len() <= 1, "Only support one node one output yet!");
            if operands.is_empty() {
                continue;
            }
            assert!(operands.len() == 1, "Only support one node one output yet!");
            let operand = &operands[0];
            let runtime_operator = &operators[i];
            let operand_shapes = &operand.borrow().shape;
            //得到需要初始化的空间
            let output_tensors = &mut runtime_operator.borrow_mut().output_operands;
            check_shape(operand_shapes);

            match output_tensors {
                Some(output_tensors_) => {
                    //如果输出空间不为空
                    Self::check_and_reshape_tensors(output_tensors_, operand);
                }
                None => {
                    // 如果输出空间没有被初始化过
                    *output_tensors = Self::init_output_tensors(operand);
                }
            }
        }
    }
    pub fn check_and_reshape_tensors<A: Zero + Clone>(
        output_operand: &mut SharedRuntimeOperand<A>,
        pnnx_operand: &SharedOperand,
    ) {
        let operand_shapes = &pnnx_operand.borrow().shape;
        let batch = operand_shapes[0];
        //检查形状是否相同
        assert_eq!(batch, output_operand.borrow().datas.len());
        assert_eq!(output_operand.borrow().shapes, pnnx_operand.borrow().shape);
        println!("===={:?}", output_operand.borrow().shapes);

        // 逐批次检查输出空间的形状是否合理，如果不合理则进行reshape
        // for batch_index  in 0..batch{
        //     let output_tensor = & output_operand.borrow().datas[batch_index];
        //     let tensor_shapes = output_tensor.borrow().shapes();
        //     match operand_shapes.len() {
        //         2 => {
        //             if tensor_shapes[0]
        //             let tesnsor = create_tensor_1::<A>(operand_shapes[1]);
        //             output_operand.datas.push(tesnsor.clone());
        //         }
        //         3 =>{
        //             let tesnsor = create_tensor_2::<A>(operand_shapes[1], operand_shapes[2]);
        //             output_operand.datas.push(tesnsor.clone());
        //         }
        //         4 =>{
        //             let tesnsor = create_tensor::<A>(operand_shapes[1], operand_shapes[2], operand_shapes[3]);
        //             output_operand.datas.push(tesnsor.clone());
        //         }
        //         _ =>{
        //             panic!();
        //         }
        //     }

        // }
    }
    pub fn init_output_tensors<A: Zero + Clone>(
        pnnx_operand: &SharedOperand,
    ) -> Option<Rc<RefCell<RuntimeOperand<A>>>> {
        let mut output_operand = RuntimeOperand::<A>::new();
        output_operand.shapes = pnnx_operand.as_ref().borrow().get_shape().clone();
        output_operand.data_type = RuntimeDataType::TypeFloat32;
        output_operand.name = pnnx_operand.as_ref().borrow().name.clone() + "_output";
        let operand_shapes = &pnnx_operand.borrow().shape;
        //输出空间的初始化
        let batch = operand_shapes[0];
        for _ in 0..batch {
            match operand_shapes.len() {
                2 => {
                    let tesnsor = create_tensor_1::<A>(operand_shapes[1]);
                    output_operand.datas.push(tesnsor.clone());
                }
                3 => {
                    let tesnsor = create_tensor_2::<A>(operand_shapes[1], operand_shapes[2]);
                    output_operand.datas.push(tesnsor.clone());
                }
                4 => {
                    let tesnsor =
                        create_tensor::<A>(operand_shapes[1], operand_shapes[2], operand_shapes[3]);
                    output_operand.datas.push(tesnsor.clone());
                }
                _ => {
                    panic!();
                }
            }
        }
        Some(Rc::new(RefCell::new(output_operand)))
    }
}

#[cfg(test)]
mod test_operator {
    use super::*;
    #[test]
    fn test_new() {
        let _operator = RuntimeOperator::<f32>::new();
    }
}

#[cfg(test)]
mod test_runtime_operator_util {

    #[test]
    fn test_init_operator_input() {}
}
