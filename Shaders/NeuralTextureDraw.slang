import NNConfig;
import LinearTransform;
import ActivationFuncs;

struct VSOutput
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD;
};

struct VSInput
{
    float3 pos : POSITION; // location 0
};

[shader("vertex")]
VSOutput vertMain(VSInput input)
{
    VSOutput output;
    output.pos = float4(input.pos * float3(0.5, 1, 1) + float3(0.5, 0, 0), 1.0);
    output.uv  = input.pos.xy * 0.5 + 0.5;
    return output;
}


struct PushConstants
{
    int   imageBase;
    int   isTraining;
    int   sizes[3];
    int   totalSize;
    float learningRate;
    uint  frameSeed;
};

[[vk::push_constant]]
PushConstants pc;


[vk::binding(2, 0)]
ByteAddressBuffer weights;

//[vk::binding(5, 0)]
// Sampler2D<half4> latentTexture;

// fragment shader
//[require(spvDescriptorHeapEXT)]
[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target
{


    DescriptorHandle<Sampler2D> texture = DescriptorHandle<Sampler2D>(uint2(3028, 16));
    DescriptorHandle<Sampler2D> latentTexture = DescriptorHandle<Sampler2D>(uint2(3029, 16));

    CoopVec<float16_t, INPUT_FEATURES> vec;

    for (int i = 0; i < INPUT_FEATURES - 4; i++)
    {
        vec[i] = FrequencyEncode(input.uv)[i];
    }
    for (int i = 0; i < 4; i++)
    {
        vec[i + INPUT_FEATURES - 4] = half(latentTexture.SampleLevel(input.uv, 0)[i]);
    }
    CoopVec<float16_t, OUTPUT_NEURONS> result;

    int buffOffset = 0;

    CoopVec<float16_t, HIDDEN_NEURONS> intermediateVecs[HIDDEN_LAYERS];

    intermediateVecs[0] = LinearMatMulAdd<float16_t, HIDDEN_NEURONS, INPUT_FEATURES>(
        vec, weights, 0, buffOffset + pc.sizes[0], CoopVecMatrixLayout::TrainingOptimal, CoopVecComponentType::Float16);


    // Moving to after  the first matrix (input * hidden sized, times two for the float16)
    buffOffset += pc.sizes[0] + HIDDEN_NEURONS * 2;

    intermediateVecs[0] = LeakyReLU(intermediateVecs[0]);

    for (int i = 1; i < HIDDEN_LAYERS; i++)
    {
        intermediateVecs[i] = LinearMatMulAdd<float16_t, HIDDEN_NEURONS, HIDDEN_NEURONS>(
            intermediateVecs[i - 1], weights, buffOffset, buffOffset + pc.sizes[1],
            CoopVecMatrixLayout::TrainingOptimal, CoopVecComponentType::Float16);
        intermediateVecs[i] = LeakyReLU(intermediateVecs[i]);

        buffOffset += pc.sizes[1] + HIDDEN_NEURONS * 2;
    }

    result =
        LinearMatMulAdd<float16_t, OUTPUT_NEURONS, HIDDEN_NEURONS>(intermediateVecs[HIDDEN_LAYERS - 1],
                                                                   weights,                  // input matrix
                                                                   buffOffset,               // matrix offset in bytes
                                                                   buffOffset + pc.sizes[2], // bias offset in bytes
                                                                   CoopVecMatrixLayout::TrainingOptimal, // matrix
                                                                   CoopVecComponentType::Float16,        // matrix
        );

    // for (int i = INPUT_FEATURES - 4; i < INPUT_FEATURES; i++)
    // vec[i] = latentTexture.SampleLevel(input.uv, 0)[i] * 100000;
    return input.uv.x > 1.5 ? latentTexture.SampleLevel(input.uv, 0) : abs(float4(result[0], result[1], result[2], 1.0)); // -texture.Sample(input.uv)
}
