

//[vk_binding(1, 0)]
//ByteAddressBuffer weights;


[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain()
{
    DescriptorHandle<RWByteAddressBuffer> input = DescriptorHandle<RWByteAddressBuffer>(uint2(3024, 0));

    //USING THIS FAILS TO COMPILE
    DescriptorHandle<ByteAddressBuffer> weights   = DescriptorHandle<ByteAddressBuffer>(uint2(3025, 0));

    DescriptorHandle<RWStructuredBuffer<CoopVec<float16_t, 64>>> outputBuffer = DescriptorHandle < RWStructuredBuffer<CoopVec<float16_t, 64>>>(uint2(3026, 0));

    CoopVec<float16_t, 64> layer = coopVecLoad<64, float16_t>(input, 64 * 2);


    let output = coopVecMatMul<float16_t, 64, 64>
    (
        layer, 
        CoopVecComponentType::Float16, 
        weights, 
        0,
        CoopVecComponentType::Float16, 
        CoopVecMatrixLayout::RowMajor,
        false, 
        64 * 2
    );

    outputBuffer[0] = output;

}
