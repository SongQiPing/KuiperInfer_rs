use regex::Regex;
use std::collections::HashMap;
use super::store_zip as store_zip;
use store_zip::StoreZipReader;



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

pub struct Attribute {
    pub type_id: i32,
    pub shape: Vec<i32>,
    pub data: Vec<u8>,
}
impl Attribute {
    pub fn new() -> Self {
        Attribute {
            type_id: 0,
            shape: Vec::new(),
            data: Vec::new(),
        }
    }
    pub fn set_type_id(&mut self, type_id: i32) {
        self.type_id = type_id;
    }
    pub fn set_data(&mut self, data: Vec<u8>) {
        self.data = data;
    }
}
const F32_DATA_TYPE: i32 = 1;
const F64_DATA_TYPE: i32 = 2;
const F16_DATA_TYPE: i32 = 3;
const I32_DATA_TYPE: i32 = 4;
const I64_DATA_TYPE: i32 = 5;
const I16_DATA_TYPE: i32 = 6;
const I8_DATA_TYPE: i32 = 7;
const U8_DATA_TYPE: i32 = 8;
const BOOL_DATA_TYPE: i32 = 9;
const CP64_DATA_TYPE: i32 = 10;
const CP128_DATA_TYPE: i32 = 11;
const CP32_DATA_TYPE: i32 = 12;

fn string_to_type(s: &str) -> i32 {
    match s {
        "f32" => F32_DATA_TYPE,
        "f64" => F64_DATA_TYPE,
        "f16" => F16_DATA_TYPE,
        "i32" => I32_DATA_TYPE,
        "i64" => I64_DATA_TYPE,
        "i16" => I16_DATA_TYPE,
        "i8" => I8_DATA_TYPE,
        "u8" => U8_DATA_TYPE,
        "bool" => BOOL_DATA_TYPE,
        "cp64" => CP64_DATA_TYPE,
        "cp128" => CP128_DATA_TYPE,
        "cp32" => CP32_DATA_TYPE,
        _ => 0, // null
    }
}

const F32_SIZE: usize = 4;
const F64_SIZE: usize = 8;
const F16_SIZE: usize = 2;
const I32_SIZE: usize = 4;
const I64_SIZE: usize = 8;
const I16_SIZE: usize = 2;
const I8_SIZE: usize = 1;
const U8_SIZE: usize = 1;
const BOOL_SIZE: usize = 1;
const CP64_SIZE: usize = 8;
const CP128_SIZE: usize = 16;
const CP32_SIZE: usize = 4;

fn type_to_elemsize(type_: i32) -> usize {
    match type_ {
        1 => F32_SIZE,
        2 => F64_SIZE,
        3 => F16_SIZE,
        4 => I32_SIZE,
        5 => I64_SIZE,
        6 => I16_SIZE,
        7 => I8_SIZE,
        8 => U8_SIZE,
        9 => BOOL_SIZE,
        10 => CP64_SIZE,
        11 => CP128_SIZE,
        12 => CP32_SIZE,
        _ => 0, // null
    }
}

pub struct Operator {
    pub inputs: Vec<Rc<RefCell<Operand>>>,
    pub outputs: Vec<Rc<RefCell<Operand>>>,
    pub type_name: String,
    pub name: String,
    pub input_names: Vec<String>,
    pub params: HashMap<String, Parameter>,
    pub attrs: HashMap<String, Attribute>,
}
impl Operator {
    pub fn new(type_name: String, name: String) -> Self {
        Operator {
            type_name,
            name,
            inputs: Vec::new(),
            outputs: Vec::new(),
            input_names: Vec::new(),
            params: HashMap::new(),
            attrs: HashMap::new(),
        }
    }
    pub fn new_operator(type_name: String, name: String) -> Rc<RefCell<Operator>> {
        Rc::new(RefCell::new(Operator::new(type_name, name)))
    }
    pub fn add_input_operand(&mut self, input_operand: Rc<RefCell<Operand>>) {
        self.inputs.push(input_operand);
        self.input_names.push(String::new());
    }
    pub fn add_output_operand(&mut self, output_operand: Rc<RefCell<Operand>>) {
        self.outputs.push(output_operand);
    }
    pub fn set_input_name(&mut self, input_name: String, operand_name: String) {
        for (id, operand) in self.inputs.iter().enumerate() {
            if operand.borrow().name == operand_name {
                self.input_names[id] = input_name;
                break;
            }
        }
    }
    pub fn get_operand(&self, operand_name: String) -> Option<Rc<RefCell<Operand>>> {
        for operand in self.inputs.iter() {
            if operand.as_ref().borrow().name == operand_name {
                return Some(operand.clone());
            }
        }
        for operand in self.outputs.iter() {
            if operand.as_ref().borrow().name == operand_name {
                return Some(operand.clone());
            }
        }
        None
    }
    pub fn insert_parmas(&mut self, parmas_key: String, parmas_value: String) {
        self.params
            .insert(parmas_key, Parameter::parse_from_string(parmas_value));
    }

    pub fn get_attribute(&mut self, attribute_name: String) -> &mut Attribute {
        let attribute: &mut Attribute = self
            .attrs
            .entry(attribute_name.clone())
            .or_insert_with(|| Attribute::new());
        attribute
    }
}
use std::cell::RefCell;
use std::rc::Rc;
pub struct Operand {
    pub producer: Option<Rc<RefCell<Operator>>>,
    pub consumers: Vec<Rc<RefCell<Operator>>>,
    pub type_id: i32,
    pub shape: Vec<usize>,
    pub name: String,
    pub params: HashMap<String, Parameter>,
}

impl Operand {
    /// Creates a new [`Operand`].
    pub fn new(name: String) -> Self {
        Operand {
            producer: Option::None,
            consumers: Vec::new(),
            type_id: 0,
            shape: Vec::new(),
            name: name,
            params: HashMap::new(),
        }
    }
    pub fn new_operand(name: String) -> Rc<RefCell<Operand>> {
        Rc::new(RefCell::new(Operand::new(name)))
    }

    pub fn set_producer(&mut self, producer: Rc<RefCell<Operator>>) {
        self.producer = Some(producer);
    }
    pub fn add_consumer(&mut self, consumer: Rc<RefCell<Operator>>) {
        self.consumers.push(consumer);
    }
    pub fn set_shape(&mut self, shape: Vec<usize >) {
        self.shape = shape;
    }
    pub fn get_shape(&self) -> &Vec<usize> {
        &self.shape
    }
}

pub struct Graph {
    pub operators: Vec<Rc<RefCell<Operator>>>,
    pub operatands: Vec<Rc<RefCell<Operand>>>,
}
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::{io, u32};

fn read_magic_number(lines: &mut io::Lines<BufReader<File>>) -> u32 {
    let num_string = lines.next().expect("File is empty").unwrap();
    let magic_number = num_string
        .parse::<u32>()
        .expect("Failed to parse magic number as u32");
    magic_number
}
fn read_operator_count(lines: &mut io::Lines<BufReader<File>>) -> (u32, u32) {
    let num_string = lines.next().expect("File is empty").unwrap();
    let mut num_parts = num_string.split_whitespace();

    let operator_count: u32 = num_parts
        .next()
        .unwrap()
        .parse()
        .expect("Failed to parse operator count");
    let operand_count: u32 = num_parts
        .next()
        .unwrap()
        .parse()
        .expect("Failed to parse operand count");

    (operator_count, operand_count)
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            operators: Vec::new(),
            operatands: Vec::new(),
        }
    }

    pub fn from_pnnx(param_path: &str, bin_path: &str) -> Self {
        let mut graph = Graph::new();

        graph.load(param_path, bin_path);

        graph
    }

    pub fn get_operand(&self, operand_name: String) -> Option<Rc<RefCell<Operand>>> {
        for op in self.operatands.iter() {
            if op.borrow().name == operand_name {
                return Some(op.clone());
            }
        }
        None
    }
    pub fn new_operator(&mut self, type_name: String, name: String) -> Rc<RefCell<Operator>> {
        let op = Operator::new_operator(type_name, name);
        self.operators.push(op.clone());
        op
    }
    pub fn new_operand(&mut self, name: String) -> Rc<RefCell<Operand>> {
        let operand = Operand::new_operand(name);
        self.operatands.push(operand.clone());
        operand
    }

    pub fn load_shape(&mut self, op: &Rc<RefCell<Operator>>, key: String, value: String) {
        let operand = op.as_ref().borrow().get_operand(key).unwrap();

        let _typestr = value[value.rfind(')').map_or(0, |pos| pos + 1)..].to_string();

        let lc = value[1..value.rfind(')').map_or(0, |pos| pos)].to_string();

        let shape: Vec<usize> = lc.split(',').map(|s| s.trim().parse().unwrap()).collect();
        operand.borrow_mut().set_shape(shape);
    }
    pub fn load_input_key(
        &mut self,
        op: &Rc<RefCell<Operator>>,
        input_name: String,
        operand_name: String,
    ) {
        op.as_ref()
            .borrow_mut()
            .set_input_name(input_name, operand_name);
    }

    pub fn load_parameter(
        &mut self,
        op: &Rc<RefCell<Operator>>,
        parmas_key: String,
        parmas_value: String,
    ) {
        op.as_ref()
            .borrow_mut()
            .insert_parmas(parmas_key, parmas_value);
    }

    pub fn load_attribute(
        &mut self,
        op: &Rc<RefCell<Operator>>,
        parmas_key: String,
        parmas_value: &String,
        store_zip_reader: &mut StoreZipReader,
    ) {
        // get data type id 
        let data_type = parmas_value
            .split(')')
            .last()
            .unwrap_or_default()
            .to_string();
        let data_type_id = string_to_type(&data_type.as_str());

        // get bytesize
        let lc = parmas_value[1..parmas_value.rfind(')').unwrap()].to_string();
        let shape: Vec<u32> = lc
            .split(',')
            .map(|s| s.trim().parse::<u32>().unwrap())
            .collect();
        let mut size: usize = 1;
        for &i in &shape {
            size *= i as usize;
        }
        let byte_size = size * type_to_elemsize(data_type_id);
        
        // get filename
        let operator_name = op.as_ref().borrow().name.clone();
        let filename = operator_name + "." + &parmas_key;
        
        // get file size 
        let file_size = store_zip_reader.get_file_size(&filename).unwrap();
        if file_size == 0 {
            return;
        }
        if file_size != byte_size{
            panic!("byte_size: {} != file_size :{}", byte_size, file_size);
        }
        let data = store_zip_reader.read_file(&filename);

        let mut borrowed_op = op.as_ref().borrow_mut();
        let attribute = borrowed_op.get_attribute(parmas_key.clone());

        attribute.set_type_id(data_type_id);
        attribute.set_data(data);

    }
    pub fn load(&mut self, param_path: &str, bin_path: &str) {
        // let mut param_buff = std::io::Cursor::new(param_path);
        let input = File::open(param_path).expect("File is empty");
        let buffered = BufReader::new(input);

        let mut store_zip_reader = store_zip::StoreZipReader::from_file(bin_path);

        let mut lines: io::Lines<BufReader<File>> = buffered.lines();
        // read magic_number

        let _magic_number: u32 = read_magic_number(&mut lines);

        let (_operator_count, _operand_count) = read_operator_count(&mut lines);

        for line in lines {
            let line_str = line.unwrap();
            let mut parts = line_str.split_whitespace();

            let operator_type = parts.next().unwrap().to_string();
            let operator_name = parts.next().unwrap().to_string();
            let input_count: u32 = parts
                .next()
                .unwrap()
                .parse()
                .expect("Failed to parse input_count");
            let output_count: u32 = parts
                .next()
                .unwrap()
                .parse()
                .expect("Failed to parse input_count");
            let op = self.new_operator(operator_type, operator_name);

            for _i in 0..input_count {
                let operand_name: String = parts.next().unwrap().to_string();
                let operand = self.get_operand(operand_name).unwrap();

                operand.borrow_mut().add_consumer(op.clone());
                op.borrow_mut().add_input_operand(operand);
            }

            for _i in 0..output_count {
                let operand_name: String = parts.next().unwrap().to_string();
                let operand = self.new_operand(operand_name);

                operand.borrow_mut().set_producer(op.clone());
                op.borrow_mut().add_output_operand(operand.clone());
            }

            for part_string in parts {
                let part_string = part_string.to_string();
                let mut key_value_iter = part_string.split('=');
                let key = key_value_iter.next().unwrap_or_default().to_string();
                let value = key_value_iter.next().unwrap_or_default().to_string();

                // println!("key:{}, value:{}", &key, &value);

                if key.starts_with('@') {
                    // attribute
                    self.load_attribute(&op, key[1..].to_string(), &value, &mut store_zip_reader);
                } else if key.starts_with('$') {
                    // operand input key
                    self.load_input_key(&op, key[1..].to_string(), value);
                } else if key.starts_with('#') {
                    // operand shape
                    self.load_shape(&op, key[1..].to_string(), value);
                } else {
                    // parameter
                    self.load_parameter(&op, key, value);
                }
            }
        }
    }
}
