extern crate kuiper_infer;

#[cfg(test)]
mod test_param {
    use std::mem::discriminant;
    #[test]
    fn test_new_string_param() {
        use kuiper_infer::pnnx::Parameter;
        let string_data = Parameter::String(String::from("ss"));

        // 验证类型是否相同
        assert_eq!(
            discriminant(&string_data),
            discriminant(&Parameter::String(String::from("other_string")))
        );
        match string_data {
            Parameter::String(value) => {
                // 在这里，value 是取出的 String 值
                println!("String value: {}", value);
            }
            _ => {
                // 如果不是 String 变体，发生 panic
                panic!("Expected a String variant");
            }
        }
    }
    #[test]
    fn test_torch_load_pnnx() {}
}
#[cfg(test)]
mod test_operator {

    #[test]
    fn test_new_operator() {
        use kuiper_infer::pnnx::Operator;
        let _operator = Operator::new("pnnx.Input".to_string(), "pnnx_input_0".to_string());
    }
}
#[cfg(test)]
mod test_operand {

    #[test]
    fn test_new_operand() {
        use kuiper_infer::pnnx::Operand;
        let _operand = Operand::new("operand_name".to_string());
    }
}

#[cfg(test)]
mod test_parameter {
    use kuiper_infer::pnnx::Parameter;

    fn param_eq(param1: &Parameter, param2: &Parameter) {
        match (param1, param2) {
            (Parameter::None, Parameter::None) => {
                // 处理两个参数都是 None 的情况
                println!("Both parameters are None.");
            }
            (Parameter::Bool(b1), Parameter::Bool(b2)) if b1 == b2 => {
                // 处理两个参数都是 Bool 类型且相等的情况
                println!("Both parameters are Bool and equal.");
            }
            (Parameter::Bool(b1), Parameter::Bool(b2)) if b1 != b2 => {
                panic!("Both parameters are Bool and equal.{}, {}", b1, b2);
            }
            (Parameter::Int(i1), Parameter::Int(i2)) if i1 == i2 => {
                // 处理两个参数都是 Int 类型且相等的情况
                println!("Both parameters are Int and equal.");
            }
            (Parameter::Int(i1), Parameter::Int(i2)) if i1 != i2 => {
                // 处理两个参数都是 Int 类型且相等的情况
                panic!("Both parameters are Bool and equal.{}, {}", i1, i2);
            }
            (Parameter::Float(f1), Parameter::Float(f2)) if f1 == f2 => {
                // 处理两个参数都是 Int 类型且相等的情况
                println!("Both parameters are Float and equal.");
            }
            (Parameter::Float(f1), Parameter::Float(f2)) if f1 != f2 => {
                // 处理两个参数都是 Int 类型且相等的情况
                panic!("Both parameters are Float and equal.{}, {}", f1, f2);
            }
            (Parameter::String(s1), Parameter::String(s2)) if s1 == s2 => {
                // 处理两个参数都是 Int 类型且相等的情况
                println!("Both parameters are String and equal.");
            }
            (Parameter::String(s1), Parameter::String(s2)) if s1 != s2 => {
                // 处理两个参数都是 Int 类型且相等的情况
                panic!("Both parameters are String and equal.{}, {}", s1, s2);
            }
            (Parameter::IntList(vec1), Parameter::IntList(vec2)) if vec1 == vec2 => {
                // 处理两个参数都是 Int 类型且相等的情况
                println!("Both parameters are IntList and equal.");
            }
            (Parameter::IntList(vec1), Parameter::IntList(vec2)) if vec1 != vec2 => {
                // 处理两个参数都是 Int 类型且相等的情况
                panic!(
                    "Both parameters are IntList and equal.{:?}, {:?}",
                    vec1, vec2
                );
            }
            (Parameter::FloatList(vec1), Parameter::FloatList(vec2)) if vec1 == vec2 => {
                // 处理两个参数都是 Int 类型且相等的情况
                println!("Both parameters are Vec and equal.");
            }
            (Parameter::FloatList(vec1), Parameter::FloatList(vec2)) if vec1 != vec2 => {
                // 处理两个参数都是 Int 类型且相等的情况
                panic!(
                    "Both parameters are FloatList and equal.{:?}, {:?}",
                    vec1, vec2
                );
            }
            (Parameter::StringList(vec1), Parameter::StringList(vec2)) if vec1 == vec2 => {
                // 处理两个参数都是 Int 类型且相等的情况
                println!("Both parameters are Vec and equal.");
            }
            (_, _) => {
                // 处理其他情况，即两个参数不相等的情况

                panic!(
                    "Parameters are not equal. param1: {:?}, param2: {:?}",
                    param1.fmt(),
                    param2.fmt()
                );
                // assert!(false);
            }
        }
    }
    #[test]
    fn test_none() {
        use kuiper_infer::pnnx::Parameter;
        let param = Parameter::parse_from_string("None".to_string());
        param_eq(&param, &Parameter::None);
        let param = Parameter::parse_from_string("()".to_string());
        param_eq(&param, &Parameter::None);

        let param = Parameter::parse_from_string("[]".to_string());
        param_eq(&param, &Parameter::None);
    }
    #[test]
    fn test_param_bool() {
        use kuiper_infer::pnnx::Parameter;
        let param = Parameter::parse_from_string("False".to_string());
        param_eq(&param, &Parameter::Bool(false));
        let param = Parameter::parse_from_string("True".to_string());
        param_eq(&param, &Parameter::Bool(true));
    }
    #[test]
    fn test_param_vec_int() {
        use kuiper_infer::pnnx::Parameter;
        let param = Parameter::parse_from_string("(1,1)".to_string());

        param_eq(&param, &Parameter::IntList(vec![1, 1]));
        let param = Parameter::parse_from_string("(1,-1)".to_string());
        param_eq(&param, &Parameter::IntList(vec![1, -1]));
    }
    #[test]
    fn test_param_vec_float() {
        use kuiper_infer::pnnx::Parameter;
        let param = Parameter::parse_from_string("(1.0, 1.0)".to_string());

        param_eq(&param, &Parameter::FloatList(vec![1.0, 1.0]));
        let param = Parameter::parse_from_string("(1.0,-1.0)".to_string());
        param_eq(&param, &Parameter::FloatList(vec![1.0, -1.0]));
    }
    #[test]
    fn test_param_vec_string() {
        use kuiper_infer::pnnx::Parameter;
        let param = Parameter::parse_from_string("(xsdfs, sdfa24)".to_string());

        param_eq(
            &param,
            &Parameter::StringList(vec!["xsdfs".to_string(), "sdfa24".to_string()]),
        );
    }
    #[test]
    fn test_param_string() {
        use kuiper_infer::pnnx::Parameter;
        let param = Parameter::parse_from_string("xsdfs".to_string());

        param_eq(
            &param,
            &Parameter::String(String::from("xsdfs")),
        );
    }
    use regex::Regex;

    fn vec_i32_is_ok(value: &String) -> bool {
        let regex = Regex::new(r"^\((\d+(,\s*\d+)*)?\)$").unwrap();

        regex.is_match(value)
    }

    #[test]
    fn test_vec_i32_is_ok() {
        let input = "(1,1)";

        if vec_i32_is_ok(&input.to_string()) {
            println!("The input is a valid i32 vector.");
        } else {
            println!("The input is not a valid i32 vector.");
            assert!(false);
        }
    }
}

#[cfg(test)]
mod test_graph {
    use std::cell::RefCell;
    use std::fs::File;
    use std::io::{self, Read};
    use std::rc::Rc;

    pub fn read_file(file_path: &str) -> io::Result<String> {
        // 打开文件
        let mut file = File::open(file_path)?;

        // 创建一个字符串来存储文件内容
        let mut content = String::new();

        // 读取文件内容到字符串
        file.read_to_string(&mut content)?;

        Ok(content)
    }
    #[test]
    fn test_graph_load_pnnx() {
        use kuiper_infer::pnnx::Graph;
        let param_path = "model_file/test_linear.pnnx.param";
        let bin_path = "model_file/test_linear.pnnx.bin";
        let _graph = Graph::from_pnnx(param_path, bin_path);
    }
    #[test]
    fn test_graph_load_pnnx_yolo() {
        use kuiper_infer::pnnx::Graph;
        let param_path = "model_file/yolov5s_batch8.pnnx.param";
        let bin_path = "model_file/yolov5s_batch8.pnnx.bin";
        let _graph = Graph::from_pnnx(param_path, bin_path);
    }
    #[test]
    fn test_new_operand() {
        use kuiper_infer::pnnx::Graph;
        let mut graph = Graph::new();
        graph.new_operand("0".to_string());
    }
    #[test]
    fn test_refcell_borrowed() {
        use std::cell::RefCell;
        use std::rc::Rc;

        struct MyStruct {
            pub _data: String,
        }
        let shared_data = Rc::new(RefCell::new(MyStruct {
            _data: String::from("Hello, Rust!"),
        }));

        // let reference1 = &shared_data.as_ref().borrow().data;
        let _reference1 = shared_data.as_ref().borrow();
        // let reference2 = shared_data.borrow();
    }
    #[test]
    fn test_get_operand() {
        use kuiper_infer::pnnx::Graph;
        use kuiper_infer::pnnx::Operand;

        let mut graph = Graph::new();
        let operand1 = graph.new_operand("0".to_string());
        let _operand2 = graph.get_operand("0".to_string()).unwrap();
        // println!("{:?}", operand.borrow());
        // operand2.borrow().set_producer();
        let cloned_operand_rc = operand1.clone();

        let borrowed_operand: Rc<RefCell<Operand>> = cloned_operand_rc.clone();
        let operand_name: &String = &borrowed_operand.as_ref().borrow().name;
        assert_eq!(operand_name.clone(), "0".to_string());
    }
    #[test]
    fn test_load_shape() {
        use kuiper_infer::pnnx::Graph;


        let mut graph = Graph::new();

        let operand1 = graph.new_operand("0".to_string());

        let operatoer = graph.new_operator("pnnx.Input".to_string(), " pnnx_input_0 ".to_string());
        operatoer
            .as_ref()
            .borrow_mut()
            .add_input_operand(operand1.clone());

        graph.load_shape(&operatoer, "0".to_string(), "(1,32)f32".to_string());
        let vec: Vec<i32> = vec![1, 32];
        assert_eq!(operand1.as_ref().borrow().get_shape(), &vec);
    }
    #[test]
    fn test_load_input_key() {
        use kuiper_infer::pnnx::Graph;
        use kuiper_infer::pnnx::Operand;

        let mut graph = Graph::new();

        let operand1: Rc<RefCell<Operand>> = graph.new_operand("1".to_string());

        let operatoer = graph.new_operator("pnnx.Input".to_string(), " pnnx_input_0 ".to_string());
        operatoer
            .as_ref()
            .borrow_mut()
            .add_input_operand(operand1.clone());

        graph.load_input_key(&operatoer, "input".to_string(), "1".to_string());

        let input_name = &operatoer.as_ref().borrow().input_names[0];

        assert_eq!(input_name, "input");
    }
    #[test]
    fn test_graph_load_attribute() {
        use kuiper_infer::pnnx::Graph;
        let mut graph = Graph::new();
        graph.new_operand("0".to_string());
    }
}
