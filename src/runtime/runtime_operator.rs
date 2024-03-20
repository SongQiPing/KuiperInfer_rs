use crate::pnnx::Parameter;

use super::RuntimeAttribute;
use super::RuntimeOperand;
use super::RuntimeParameter;
use ndarray::prelude::*;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

struct Layer {}
/// 计算图中的计算节点
pub struct RuntimeOperator<A> {
    pub has_forward: bool,
    pub name: String,             // 计算节点的名称
    pub type_name: String,            // 计算节点的类型
    pub layer: Option<Rc<Layer>>, // 节点对应的计算Layer

    pub output_names: Vec<String>, // 节点的输出节点名称
    pub output_operands: Option<Rc<RefCell<RuntimeOperand<A>>>>, // 节点的输出操作数

    pub input_operands: HashMap<String, Rc<RefCell<RuntimeOperand<A>>>>, // 节点的输入操作数
    pub input_operands_seq: Vec<Rc<RefCell<RuntimeOperand<A>>>>, // 节点的输入操作数，顺序排列
    pub output_operators: HashMap<String, Rc<RefCell<RuntimeOperator<A>>>>, // 输出节点的名字和节点对应

    pub params: HashMap<String, Rc<RefCell<Parameter>>>, // 算子的参数信息
    pub attribute: HashMap<String, Rc<RefCell<RuntimeAttribute>>>, // 算子的属性信息，内含权重信息
}

impl<A> RuntimeOperator<A> {
    pub fn new() -> Self{
        RuntimeOperator{
            has_forward:false, 
            name:String::new(),
            type_name:String::new(),
            layer:Some(Rc::new(Layer{})), 

            output_names: Vec::new(),
            output_operands: None, 
            input_operands:HashMap::new(),
            input_operands_seq:Vec::new(),
            output_operators:HashMap::new(),
            params:HashMap::new(),
            attribute:HashMap::new(),
        }

    }
}
#[cfg(test)]
mod test_operator {
    use super::*;
    #[test]
    fn test_new(){
        let operator =  RuntimeOperator::<f32>::new();
    }

}