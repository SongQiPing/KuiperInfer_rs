use super::RuntimeDataType;
use num_traits::Zero;
use crate::data::SharedTensor;
pub struct RuntimeOperand<A> {
    pub name: String,                       // 操作数的名称
    pub shapes: Vec<usize>,                 // 操作数的形状
    pub datas: Vec<SharedTensor<A>>, // 存储操作数
    pub data_type: RuntimeDataType,         // 操作数的类型，一般是float
}

impl<A> RuntimeOperand<A> 
where
    A: Clone + Zero,
{
    pub fn new() -> Self {
        RuntimeOperand {
            name: String::new(),
            shapes: Vec::new(),
            datas: Vec::new(),
            data_type: RuntimeDataType::TypeUnknown,
        }
    }
}
