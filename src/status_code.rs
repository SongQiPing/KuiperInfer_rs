#[derive(Debug, Clone)]
pub enum StatusCode {
    UnknownCode = -1,
    Success=0,
    InferInputEmpty=1,
    InferOutputsEmpty = 2,
    InferParameterError = 3,
    InferDimMismatch = 4,
  
    FunctionNotImplement = 5,
    ParseWeightError = 6,
    ParseParameterError = 7,
    ParseNullOperator = 8,

}
pub enum ParseParameterAttrStatus {
    ParameterMissingUnknown,
    ParameterMissingStride,
    ParameterMissingPadding,
    ParameterMissingKernel,
    ParameterMissingUseBias,
    ParameterMissingInChannel,
    ParameterMissingOutChannel,
  
    ParameterMissingEps,
    ParameterMissingNumFeatures,
    ParameterMissingDim ,
    ParameterMissingExpr ,
    ParameterMissingOutHW ,
    ParameterMissingShape,
    ParameterMissingGroups ,
    ParameterMissingScale,
    ParameterMissingResizeMode,
    ParameterMissingDilation,
    ParameterMissingPaddingMode ,
  
    AttrMissingBias,
    AttrMissingWeight,
    AttrMissingRunningMean,
    AttrMissingRunningVar,
    AttrMissingOutFeatures ,
    AttrMissingYoloStrides ,
    AttrMissingYoloAnchorGrides,
    AttrMissingYoloGrides,
  
    ParameterAttrParseSuccess
}