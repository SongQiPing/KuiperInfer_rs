use super::RuntimeDataType;
use crate::data::Tensor;
use ndarray::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

pub struct RuntimeOperand<A> {
    pub name: String,                       // 操作数的名称
    pub shapes: Vec<usize>,                 // 操作数的形状
    pub datas: Vec<Rc<RefCell<Tensor<A>>>>, // 存储操作数
    pub data_type: RuntimeDataType,         // 操作数的类型，一般是float
}

impl<A> RuntimeOperand<A> {
    pub fn new() -> Self {
        RuntimeOperand {
            name: String::new(),
            shapes: Vec::new(),
            datas: Vec::new(),
            data_type: RuntimeDataType::TypeUnknown,
        }
    }
}
