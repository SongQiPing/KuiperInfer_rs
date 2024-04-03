pub mod layer;
pub use layer::Layer;
pub use layer::LayerError;
pub use layer::ParameterData;
pub use layer::WeightData;

pub use layer::ParameteraGetterSetter;
pub use layer::RuntimeOperatorData;

// pub use layer::NonParamLayer;

pub mod layer_factory;
pub use layer_factory::LayerRegisterer;
pub mod layer_factory1;
