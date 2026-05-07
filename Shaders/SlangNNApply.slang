// import NeuralShading_Shaders.CooperativeVectorAutoDiff;

struct MnistDatapoint
{
    float16_t img[28][28];
    uint16_t  label;
}


[vk::binding(0, 0)]
RWByteAddressBuffer input; // [1 2 3 4 …]
[vk::binding(0, 0)]
RWStructuredBuffer<MnistDatapoint> mnistData;

[vk::binding(1, 0)]
RWStructuredBuffer<float16_t> outputBuffer;

[vk::binding(2, 0)]
RWByteAddressBuffer weights;
[vk::binding(2, 0)]
RWStructuredBuffer<float16_t> weightsFP16;

[vk::binding(3, 0)]
RWByteAddressBuffer weightAdjustment;
[vk::binding(3, 0)]
RWStructuredBuffer<float16_t> weightAdjustmentFP16;

struct PushConstants
{
    int   imageBase;
    int   isTraining;
    int   sizes[3];
    int   totalSize;
    float learningRate;
}
[[vk::push_constant]]
PushConstants pc;

[vk::binding(5, 0)]
RWTexture2D<half4> latentTexture;

[vk::binding(7, 0)]
RWStructuredBuffer<float> latentTrainingBuffer;

[vk::binding(8, 0)]
RWStructuredBuffer<float> adamTrainingBuffer;




[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 threadID: SV_DispatchThreadID)
{
    float b1          = 0.9;
    float b2          = 0.999;
    float epsilon     = 1e-8;

    float b1_divisor = 1.f / (1 - pow(b1, pc.imageBase));

    float b2_divisor = 1.f / (1 - pow(b2, pc.imageBase));

    int weightCount = (pc.totalSize) / 2;
    if (threadID.x < weightCount)
    {


        // DescriptorHandle<RWStructuredBuffer<float>> adamTrainingBuffer = DescriptorHandle<RWStructuredBuffer<float>>(uint2(3032, 0));
        adamTrainingBuffer[threadID.x * 2] = lerp(weightAdjustmentFP16[threadID.x], adamTrainingBuffer[threadID.x * 2], b1);


        //this float cast is load bearing, I wouldn't know why unfortunately.
        adamTrainingBuffer[threadID.x * 2 + 1] = lerp(pow(float(weightAdjustmentFP16[threadID.x]), 2), adamTrainingBuffer[threadID.x * 2 + 1], b2);



        float mD = adamTrainingBuffer[threadID.x * 2] * b1_divisor;

        float vD = adamTrainingBuffer[threadID.x * 2 + 1] * b2_divisor;

        weightsFP16[threadID.x] += pc.learningRate * mD / (sqrt(vD) + epsilon);
        weightAdjustmentFP16[threadID.x] = 0;
    }
    //DescriptorHandle<RWTexture2D<half4>> latentTexture         = DescriptorHandle<RWTexture2D<half4>>(uint2(3029, 0));
    //DescriptorHandle<RWTexture2D<half4>> latentTrainingTexture = DescriptorHandle<RWTexture2D<half4>>(uint2(3030, 16));

    //DescriptorHandle<RWStructuredBuffer<float>> latentTrainingBuffer =
    //    DescriptorHandle<RWStructuredBuffer<float>>(uint2(3031, 0));

    int2 dimensions;
    latentTexture.GetDimensions(dimensions.x, dimensions.y);

    int pixelCount = dimensions.x * dimensions.y;

    if (threadID.x >= weightCount && threadID.x < weightCount + pixelCount)
    {
        var   pixIndex    = threadID.x - weightCount;
        uint2 imageCoords = uint2(pixIndex % dimensions.x, pixIndex / dimensions.x);

        float4 delta = float4(1);
        for (int i = 0; i < 4; i++)
        {
            adamTrainingBuffer[weightCount * 2 + pixIndex * 8 + 2 * i] = lerp( latentTrainingBuffer[pixIndex * 4 + i], adamTrainingBuffer[weightCount * 2 + pixIndex * 8 + 2 * i], b1);

            adamTrainingBuffer[weightCount * 2 + pixIndex * 8 + 2 * i + 1] = lerp(pow(float(latentTrainingBuffer[pixIndex * 4 + i]), 2), adamTrainingBuffer[weightCount * 2 + pixIndex * 8 + 2 * i + 1], b2);

            float mD = adamTrainingBuffer[weightCount * 2 + pixIndex * 8 + 2 * i] * b1_divisor;

            float vD = adamTrainingBuffer[weightCount * 2 + pixIndex * 8 + 2 * i + 1] * b2_divisor;


            delta[i] = pc.learningRate * mD / (sqrt(vD) + epsilon);

            latentTrainingBuffer[pixIndex * 4 + i] = 0;
        }

        latentTexture[imageCoords] += delta;

    }
}
