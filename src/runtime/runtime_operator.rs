use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::cell::RefCell;

// /// 计算图中的计算节点
// struct RuntimeOperator {
//     has_forward: bool,
//     name: String,                    /// 计算节点的名称
//     type_: String,                   /// 计算节点的类型
//     layer: Option<Rc<Layer>>,        /// 节点对应的计算Layer

//     output_names: Vec<String>,       /// 节点的输出节点名称
//     output_operands: Option<Rc<Ref <RuntimeOperand>>>,  /// 节点的输出操作数

//     input_operands: HashMap<String, Rc<Ref <RuntimeOperand>>>,  /// 节点的输入操作数
//     input_operands_seq: Vec<Rc<Ref <RuntimeOperand>>>,           /// 节点的输入操作数，顺序排列
//     output_operators: HashMap<String, Rc<RefCell<RuntimeOperator>>>,  /// 输出节点的名字和节点对应

//     params: HashMap<String, Rc<RuntimeParameter>>,         /// 算子的参数信息
//     attribute: HashMap<String, Rc<RuntimeAttribute>>,       /// 算子的属性信息，内含权重信息
// }
