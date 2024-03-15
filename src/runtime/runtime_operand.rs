use ndarray::prelude::*;
use std::rc::Rc;
use std::cell::RefCell;
use crate::data::Tensor;
use super::RuntimeDataType;


pub struct RuntimeOperand<A>{
    name: String,                   // 操作数的名称
    shapes: Vec<usize>,               // 操作数的形状
    datas: Vec<Rc<RefCell <Tensor<A>>>>,    // 存储操作数
    data_type: RuntimeDataType,        // 操作数的类型，一般是float
}


