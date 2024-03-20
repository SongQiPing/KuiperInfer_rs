use std::vec::Vec;

use crate::runtime::RuntimeDataType;
use std::mem;

pub struct RuntimeAttribute {
    pub weight_data: Vec<u8>,   // 节点中的权重参数
    pub shape: Vec<i32>,        // 节点中的形状信息
    pub dtype: RuntimeDataType, // 节点中的数据类型
}

impl RuntimeAttribute {
    pub fn get<T>(&mut self, need_clear_weight: bool) -> Vec<T>
    where
        T: Copy,
    {
        // 检查节点属性中的权重类型
        assert!(!self.weight_data.is_empty());
        if let RuntimeDataType::TypeUnknown = self.dtype {
            panic!("dtype is TypeUnknown");
        }

        let mut weights: Vec<T> = Vec::new();
        let float_size = mem::size_of::<T>();
        assert_eq!(self.weight_data.len() % float_size, 0);

        // Assuming self.weight_data is a Vec<u8> containing raw bytes of weights
        for i in (0..self.weight_data.len()).step_by(float_size) {
            let raw_data: Vec<u8> = self.weight_data[i..i + float_size].to_vec();
            let value: T = unsafe { *(raw_data.as_ptr() as *const T) };
            weights.push(value);
        }

        if need_clear_weight {
            // Clear weight data if needed
            self.weight_data.clear();
        }

        weights
    }
    pub fn clear_weight(& mut self){
        self.weight_data.clear();

    }
}

#[cfg(test)]
mod test_runtime_attribute {

    use super::*;
    #[test]
    fn test_get_with_float32() {
        let mut runtime_attr = RuntimeAttribute {
            weight_data: vec![0, 0, 128, 63], // example bytes for f32: 1.0
            shape: vec![],
            dtype: RuntimeDataType::TypeFloat32,
        };
        let weights: Vec<f32> = runtime_attr.get(false);
        assert_eq!(weights, vec![1.0]);

        println!("{:?}", weights); // Output: [1.0]
    }
    #[test]
    fn test_get_with_u8() {
        let mut runtime_attr = RuntimeAttribute {
            weight_data: vec![1, 2, 3, 4],
            shape: vec![],
            dtype: RuntimeDataType::TypeFloat32,
        };

                let bytes: Vec<u8> = runtime_attr.get(false);
        assert_eq!(bytes, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_get_with_u32() {
        let mut runtime_attr = RuntimeAttribute {
            weight_data: vec![0x01, 0x02, 0x03, 0x04], // Little-endian representation of the u32 value 0x04030201
            shape: vec![],
            dtype: RuntimeDataType::TypeFloat32,
        };

        let values: Vec<u32> = runtime_attr.get(false);
        assert_eq!(values, vec![0x04030201]);
    }
}
