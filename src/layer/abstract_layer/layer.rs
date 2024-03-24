use crate::runtime::RuntimeOperator;
use std::cell::RefCell;
use std::rc::Rc;
use crate::data::SharedTensor;
pub enum LayerError {
    LocalFileHeaderInvalid,
    CentralDirectoryFileHeaderInvalid,
    EndOfCentralDirectoryInvalid,
    CantSeekToDirEnd,
    CantSeekToFileHeader,
    CantSeekSkip,
    ParseError,
    ReadStringError,
    CantSeekToDirStart,
    UnsupportedCompressionMethod,
    DecompressionError,
    DataReadError,
}
pub trait Layer<A> {
    fn forward(&self) -> Result<(), LayerError>;
    fn forward_with_tensors(
        &self,
        inputs: &SharedTensor<A>,
        outputs: &SharedTensor<A>,
    ) -> Result<(), LayerError>;
    fn weights(&self) -> Result<&Vec<SharedTensor<A>>, LayerError>;
    fn bias(&self) -> Result<&Vec<SharedTensor<A>>, &Vec<SharedTensor<A>>>;
    fn set_weights(&mut self, weights: &Vec<SharedTensor<A>>);
    fn set_bias(&mut self, bias: &Vec<SharedTensor<A>>);
    fn set_weights_from_floats(&mut self, weights: &[A]);
    fn set_bias_from_floats(&mut self, bias: &[A]);
    fn layer_name(&self) -> &String;
    fn set_runtime_operator(&mut self, runtime_operator: Rc<RefCell<RuntimeOperator<A>>>);
}

trait NonParamLayer<A>: Layer<A> {}

#[cfg(test)]
mod test_abrastra_layer {
    #[test]
    fn test_new() {
        let _layer_name = "abstra_layer".to_string();
        // let abrastra_layer = AbrastraLayer::<f32>::new(& layer_name);
    }

    #[test]
    fn test_forward() {}
}
