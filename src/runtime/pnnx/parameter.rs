use regex::Regex;

#[derive(Clone, Debug)]
pub enum Parameter {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    IntList(Vec<i32>),
    FloatList(Vec<f32>),
    StringList(Vec<String>),
}
fn vec_i64_is_ok(value: &String) -> bool {
    let regex = Regex::new(r"^\((-?\d+(,\s*-?\d+)*)?\)$").unwrap();

    regex.is_match(value)
}
fn vec_f64_is_ok(value: &String) -> bool {
    let regex = Regex::new(r"^\((-?\d+(\.\d+)?(,\s*-?\d+(\.\d+)?)*)?\)$").unwrap();

    regex.is_match(value)
}
impl Parameter {
    /// 获取参数为布尔值的值。
    ///
    /// 如果参数不是布尔值类型，将返回 `None`。
    pub fn get_bool(&self) -> Option<bool> {
        match self {
            Parameter::Bool(value) => Some(*value),
            _ => None,
        }
    }

    /// 获取参数为整数的值。
    ///
    /// 如果参数不是整数类型，将返回 `None`。
    pub fn get_int(&self) -> Option<i64> {
        match self {
            Parameter::Int(value) => Some(*value),
            _ => None,
        }
    }

    /// 获取参数为浮点数的值。
    ///
    /// 如果参数不是浮点数类型，将返回 `None`。
    pub fn get_float(&self) -> Option<f64> {
        match self {
            Parameter::Float(value) => Some(*value),
            _ => None,
        }
    }

    /// 获取参数为字符串的值。
    ///
    /// 如果参数不是字符串类型，将返回 `None`。
    pub fn get_string(&self) -> Option<String> {
        match self {
            Parameter::String(value) => Some(value.clone()),
            _ => None,
        }
    }

    /// 获取参数为整数列表的值。
    ///
    /// 如果参数不是整数列表类型，将返回 `None`。
    pub fn get_int_list(&self) -> Option<Vec<i32>> {
        match self {
            Parameter::IntList(value) => Some(value.clone()),
            _ => None,
        }
    }

    /// 获取参数为浮点数列表的值。
    ///
    /// 如果参数不是浮点数列表类型，将返回 `None`。
    pub fn get_float_list(&self) -> Option<Vec<f32>> {
        match self {
            Parameter::FloatList(value) => Some(value.clone()),
            _ => None,
        }
    }

    /// 获取参数为字符串列表的值。
    ///
    /// 如果参数不是字符串列表类型，将返回 `None`。
    pub fn get_string_list(&self) -> Option<Vec<String>> {
        match self {
            Parameter::StringList(value) => Some(value.clone()),
            _ => None,
        }
    }
}
impl Parameter {
    /// 从字符串解析为参数。
    ///
    /// 此函数接受一个字符串 `value` 作为输入，并将其解析为一个参数对象。参数类型表示系统中的某个参数，但此处不具体指定参数的性质。
    ///
    /// # 参数
    ///
    /// * `value` - 包含要解析为参数的值的字符串。
    ///
    /// # 返回值
    ///
    /// 从输入字符串解析而来的参数对象。
    ///
    /// # 示例
    ///
    /// ```
    /// use kuiper_infer::pnnx::Parameter;
    ///
    /// let param_str = "(1, 2)"; // 示例输入字符串
    /// let parameter = Parameter::parse_from_string(param_str.to_string());
    /// assert_eq!(parameter.get_int_list().unwrap(), vec![1, 2]); // 假设 Parameter 有一个返回解析值的 `value()` 方法
    /// ```
    pub fn parse_from_string(value: String) -> Parameter {
        if value == "None" || value == "()" || value == "[]" {
            return Parameter::None;
        } else if value == "True" || value == "False" {
            return Parameter::Bool(value == "True");
        } else if value.parse::<i64>().is_ok() {
            return Parameter::Int(value.parse().unwrap());
        } else if value.parse::<f64>().is_ok() {
            // 如果可以解析为浮点数，则返回 Float 类型的 Parameter
            return Parameter::Float(value.parse().unwrap());
        } else if vec_i64_is_ok(&value) {
            // 如果以 "[" 开头且以 "]" 结尾，则尝试解析为 IntList 类型的 Parameter
            let inner_values: Vec<i32> = value[1..value.len() - 1]
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();

            return Parameter::IntList(inner_values);
        } else if vec_f64_is_ok(&value) {
            // 如果以 "[" 开头且以 "]" 结尾，则尝试解析为 FloatList 类型的 Parameter
            let inner_values: Vec<f32> = value[1..value.len() - 1]
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();

            return Parameter::FloatList(inner_values);
        } else if value.starts_with("(") && value.ends_with(")") {
            let inner_values: Vec<String> = value[1..value.len() - 1]
                .split(',')
                .map(|s| s.trim().to_string())
                .collect();
            return Parameter::StringList(inner_values);
        } else {
            let inner_values = value.to_string();

            return Parameter::String(inner_values);
        }
    }
    /// 格式化参数值为字符串。
    pub fn fmt(&self) -> String {
        match self {
            Parameter::None => "None".to_string(),
            Parameter::Bool(b) => {
                if *b {
                    "True".to_string()
                } else {
                    "False".to_string()
                }
            }
            Parameter::Int(i) => i.to_string(),
            Parameter::Float(f) => f.to_string(),
            Parameter::String(s) => s.to_string(),
            Parameter::IntList(list) => {
                let inner: Vec<String> = list.iter().map(|&i| i.to_string()).collect();
                format!("[{}]", inner.join(", "))
            }
            Parameter::FloatList(list) => {
                let inner: Vec<String> = list.iter().map(|&f| f.to_string()).collect();
                format!("[{}]", inner.join(", "))
            }
            Parameter::StringList(list) => {
                format!("[{}]", list.join(", "))
            }
        }
    }
}

#[cfg(test)]
mod test_parameter {
    #[test]
    fn test_parse_from_string() {
        use crate::pnnx::Parameter;

        let param_str = "(1, 2)"; // 示例输入字符串
        let parameter = Parameter::parse_from_string(param_str.to_string());
        assert_eq!(parameter.get_int_list().unwrap(), vec![1, 2]);
    }
    #[test]
    fn test_get_bool() {
        use crate::pnnx::Parameter;
        let param_str = "True";
        let parameter = Parameter::parse_from_string(param_str.to_string());
        assert_eq!(parameter.get_bool().unwrap(), true);

        let param_str = "False";
        let parameter = Parameter::parse_from_string(param_str.to_string());
        assert_eq!(parameter.get_bool().unwrap(), false);
    }
    #[test]
    fn test_get_int() {
        use crate::pnnx::Parameter;
        let param_str = "1010";
        let parameter = Parameter::parse_from_string(param_str.to_string());
        assert_eq!(parameter.get_int().unwrap(), 1010);
    }
    #[test]
    fn test_get_float() {
        use crate::pnnx::Parameter;
        let param_str = "1010.0";
        let parameter = Parameter::parse_from_string(param_str.to_string());
        assert_eq!(parameter.get_float().unwrap(), 1010.0);
    }

    #[test]
    fn test_get_string() {
        use crate::pnnx::Parameter;
        let param_str = "asbse s";
        let parameter = Parameter::parse_from_string(param_str.to_string());
        assert_eq!(parameter.get_string().unwrap(), "asbse s".to_string());
    }
    #[test]
    fn test_get_int_list() {
        use crate::pnnx::Parameter;
        let param_str = "(1,2, 3)";
        let parameter = Parameter::parse_from_string(param_str.to_string());
        assert_eq!(parameter.get_int_list().unwrap(), vec![1, 2, 3]);
    }
    #[test]
    fn test_get_float_list() {
        use crate::pnnx::Parameter;
        let param_str = "(1.0,2.0, 3.0)";
        let parameter = Parameter::parse_from_string(param_str.to_string());
        assert_eq!(parameter.get_float_list().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_get_string_list() {
        use crate::pnnx::Parameter;
        let param_str = "(text1,text2, text3)";
        let parameter = Parameter::parse_from_string(param_str.to_string());
        assert_eq!(
            parameter.get_string_list().unwrap(),
            vec![
                "text1".to_string(),
                "text2".to_string(),
                "text3".to_string()
            ]
        );
    }
}
