use super::RuntimeAttribute;
use super::RuntimeOperand;
use super::RuntimeParameter;
use ndarray::prelude::*;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

struct Layer {}
/// 计算图中的计算节点
pub struct RuntimeOperator<A, D: Dimension> {
    has_forward: bool,
    name: String,             // 计算节点的名称
    type_: String,            // 计算节点的类型
    layer: Option<Rc<Layer>>, // 节点对应的计算Layer

    output_names: Vec<String>, // 节点的输出节点名称
    output_operands: Option<Rc<RefCell<RuntimeOperand<A, D>>>>, // 节点的输出操作数

    input_operands: HashMap<String, Rc<RefCell<RuntimeOperand<A, D>>>>, // 节点的输入操作数
    input_operands_seq: Vec<Rc<RefCell<RuntimeOperand<A, D>>>>, // 节点的输入操作数，顺序排列
    output_operators: HashMap<String, Rc<RefCell<RuntimeOperator<A, D>>>>, // 输出节点的名字和节点对应

    params: HashMap<String, Rc<RefCell<RuntimeParameter>>>, // 算子的参数信息
    attribute: HashMap<String, Rc<RefCell<RuntimeAttribute>>>, // 算子的属性信息，内含权重信息
}


#[cfg(test)]
mod test_operator {

}