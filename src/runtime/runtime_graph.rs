use ndarray::prelude::*;
use num_traits::Zero;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use super::pnnx;
use super::RuntimeAttribute;
use super::RuntimeOperand;
use super::RuntimeOperator;
use super::RuntimeParameter;

pub struct RuntimeGraph<A, D: Dimension> {
    intput_name: String, //计算输入节点的名称
    output_name: String, //计算图输入节点的名称
    param_path: String,  //计算图的结构文件
    bin_path: String,    //计算图的权重文件

    operators: Vec<Rc<RefCell<RuntimeOperator<A, D>>>>,
    operators_maps: HashMap<String, Rc<RefCell<RuntimeOperator<A, D>>>>,

    graph: Rc<RefCell<pnnx::Graph>>,
}

impl<A, D> RuntimeGraph<A, D>
where
    A: Clone + Zero,
    D: Dimension,
{
    pub fn new(param_path: String, bin_path: String) -> Self {
        RuntimeGraph {
            intput_name: String::new(),
            output_name: String::new(),
            param_path: param_path,
            bin_path: bin_path,
            operators: Vec::new(),
            operators_maps: HashMap::new(),
            graph: Rc::new(RefCell::new(pnnx::Graph::new() )),
        }
    }
    pub fn set_bin_path(& mut self, bin_path:String){
        self.bin_path = bin_path;

    }

    pub fn set_param_path(&mut self, param_path:String){
        self.param_path = param_path;
    }

    pub fn get_bin_path(& self) -> & String{
        & self.bin_path

    }    
    pub fn get_param_path(& self) -> & String{
        & self.param_path
    }
    pub fn init(& mut self){
        if self.bin_path.is_empty() || self.param_path.is_empty(){
            panic!("The bin path or param path is empty");

        }

    }



}

#[cfg(test)]
mod test_runrime_graph {    
    use super::*;
    #[test]
    fn test_new_graph(){
        let param_path = "model_file/test_linear.pnnx.param".to_string();
        let bin_path = "model_file/test_linear.pnnx.bin".to_string();
        // let runtime_grpah: RuntimeGraph<f32, _> = RuntimeGraph::<f32, _>::new(param_path, bin_path);

    }
}