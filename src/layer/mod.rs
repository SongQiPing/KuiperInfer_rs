pub mod abstract_layer;

pub use abstract_layer::layer::Layer;
pub use abstract_layer::layer::LayerError;
pub use abstract_layer::layer::ParameterData;
pub use abstract_layer::layer::WeightData;

pub use abstract_layer::layer_factory::LayerRegisterer;
pub mod details;
