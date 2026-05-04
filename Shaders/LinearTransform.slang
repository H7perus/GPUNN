


CoopVec<T, M> LinearMatMul<T : __BuiltinFloatingPointType, let M : int, let K : int>(
    CoopVec<T, K> input, 
    ByteAddressBuffer matrixBuffer, 
    uint matrixOffset, 
    constexpr CoopVecMatrixLayout matrixLayout, 
    constexpr CoopVecComponentType componentType)
{
    return coopVecMatMul<T, M>(
        input,
        componentType, 
        matrixBuffer, 
        matrixOffset,  
        componentType, 
        matrixLayout, 
        false, 
        0
    );
}

CoopVec<T, K> LinearMatMul_Backwards<T : __BuiltinFloatingPointType, let M : int, let K : int>(
    CoopVec<T, K> input,
    CoopVec<T, M> gradient,
    ByteAddressBuffer matrixBuffer,
    RWByteAddressBuffer matrixDerivativeBuffer,
    uint matrixOffset, 
    constexpr CoopVecMatrixLayout matrixLayout, 
    constexpr CoopVecComponentType componentType)
{
    coopVecOuterProductAccumulate(gradient, input, matrixDerivativeBuffer, matrixOffset, 0, matrixLayout, componentType);

    return coopVecMatMul<T, K>(
        gradient,
        componentType, 
        matrixBuffer, 
        matrixOffset,  
        componentType, 
        matrixLayout,
        true, 
        0
    );
}



CoopVec<T, M> LinearMatMulAdd<T : __BuiltinFloatingPointType, let M : int, let K : int>(
    CoopVec<T, K> input, 
    ByteAddressBuffer matrixBiasBuffer, 
    uint matrixOffset, 
    uint biasOffset,
    constexpr CoopVecMatrixLayout matrixLayout, 
    constexpr CoopVecComponentType componentType)
{
    return coopVecMatMulAdd<T, M>(
        input,
        componentType, 
        matrixBiasBuffer, 
        matrixOffset, 
        componentType, 
        matrixBiasBuffer,
        biasOffset, 
        componentType, 
        matrixLayout, 
        false, 
        0
    );
}

CoopVec<T, K> LinearMatMulAdd_Backwards<T : __BuiltinFloatingPointType, let M : int, let K : int>(
    CoopVec<T, K> input,
    CoopVec<T, M> gradient,
    ByteAddressBuffer matrixBiasBuffer,
    RWByteAddressBuffer matrixBiasDerivativeBuffer,
    uint matrixOffset, 
    uint biasOffset,
    constexpr CoopVecMatrixLayout matrixLayout, 
    constexpr CoopVecComponentType componentType)
{
    coopVecOuterProductAccumulate(gradient, input, matrixBiasDerivativeBuffer, matrixOffset, 0, matrixLayout, componentType);
    coopVecReduceSumAccumulate(gradient, matrixBiasDerivativeBuffer, biasOffset);

    return coopVecMatMul<T, K>(
        gradient,
        componentType, 
        matrixBiasBuffer, 
        matrixOffset,  
        componentType, 
        matrixLayout, 
        true, 
        0
    );
}


