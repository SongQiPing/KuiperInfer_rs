#[derive(Debug)]
pub enum RuntimeDataType {
    TypeUnknown = 0,
    TypeFloat32 = 1,
    TypeFloat64 = 2,
    TypeFloat16 = 3,
    TypeInt32 = 4,
    TypeInt64 = 5,
    TypeInt16 = 6,
    TypeInt8 = 7,
    TypeUInt8 = 8,
}