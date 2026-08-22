
// import NeuralShading_Shaders.CooperativeVectorAutoDiff;
import NNConfig;
import ActivationFuncs;
import LinearTransform;

struct MnistDatapoint
{
    float16_t img[28 * 28];
    uint16_t  label;
    uint16_t  padd[7];
}


uint pcg_hash(uint seed)
{
    uint state = seed * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// uint -> [0, 1)
float uintToUnitFloat(uint x)
{
    return float(x) * (1.0 / 4294967296.0); // 1 / 2^32
}

// Mix two seeds (boost-style splice)
uint hashCombine(uint a, uint b)
{
    return pcg_hash(a ^ (b + 0x9e3779b9u + (a << 6) + (a >> 2)));
}

// Stateful generator — advances each call
struct PRNG
{
    uint state;

    __init(uint seed)
    {
        // Hash once on init so state==0 isn't a fixed point
        state = pcg_hash(seed);
    }

    [mutating]
    uint nextUint()
    {
        state = pcg_hash(state);
        return state;
    }
    [mutating]
    float nextFloat()
    {
        return uintToUnitFloat(nextUint());
    }
    [mutating]
    float2 nextFloat2()
    {
        return float2(nextFloat(), nextFloat());
    }
    [mutating]
    float3 nextFloat3()
    {
        return float3(nextFloat(), nextFloat(), nextFloat());
    }
};




[vk::binding(0, 0)]
ByteAddressBuffer weights;

[vk::binding(1, 0)]
RWByteAddressBuffer weightAdjustment;
[vk::binding(1, 0)]
RWStructuredBuffer<float16_t> weightAdjustmentFP16;


struct PushConstants
{
    int   imageBase;
    int   isTraining;
    int   sizes[3];
    int   totalSize;
    float learningRate;
    uint  frameSeed;
    //int   batchSize;
}

[[vk::push_constant]]
PushConstants pc;

[vk::binding(2, 0)]
Sampler2D texture;

//[vk::binding(3, 0)]
//Sampler2D latentTexture;

[vk::binding(4, 0)]
RWStructuredBuffer<Atomic<float>> latentTrainingBuffer;







const static int batch_size = 4096;





[shader("compute")]
[numthreads(128, 1, 1)]
void computeMain(uint3 threadID: SV_DispatchThreadID)
{

    float test;

    half test2 = half(test);

    int imageIndex = threadID.x + pc.imageBase * batch_size;

    // if (imageIndex >= 60000 + 10000 * (1 - pc.isTraining))
    //     return;
    // DescriptorHandle<Sampler2D>          latentTexture         = DescriptorHandle<Sampler2D>(uint2(3029, 16));

    // DescriptorHandle<RWStructuredBuffer<Atomic<float>>> latentTrainingBuffer =
    // DescriptorHandle<RWStructuredBuffer<Atomic<float>>>(uint2(3031, 0));

    DescriptorHandle<Sampler2D> latentTexture = DescriptorHandle<Sampler2D>(uint2(3027, 16));

    uint seed = hashCombine(imageIndex, pc.frameSeed);
    PRNG rng  = PRNG(seed);

    float2 randomUV;
    randomUV = rng.nextFloat2();

    uint2 coords = uint2(threadID.x % (uint)sqrt(batch_size), threadID.x / (uint)sqrt(batch_size));
    randomUV     = float2(coords) / sqrt(batch_size) + randomUV / sqrt(batch_size);

    // snap UV to a pixel:
    randomUV = int2(randomUV * 256)  / 256.f;

    // randomUV = min(max(randomUV, 0), 1.0);

    CoopVec<float16_t, INPUT_FEATURES> vec;
    for (int i = 0; i < INPUT_FEATURES - 4; i++)
    {
        vec[i] = FrequencyEncode(randomUV)[i];
    }
    for (int i = 0; i < 4; i++)
    {
        vec[i + INPUT_FEATURES - 4] = half(latentTexture.SampleLevel(randomUV, 0)[i]);
    }

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


    CoopVec<float16_t, OUTPUT_NEURONS> result =
        LinearMatMulAdd<float16_t, OUTPUT_NEURONS, HIDDEN_NEURONS>(intermediateVecs[HIDDEN_LAYERS - 1],
                                                                   weights,                  // input matrix
                                                                   buffOffset,               // matrix offset in bytes
                                                                   buffOffset + pc.sizes[2], // bias offset in bytes
                                                                   CoopVecMatrixLayout::TrainingOptimal, // matrix
                                                                   CoopVecComponentType::Float16,        // matrix
        );

    int outputOffset = imageIndex * OUTPUT_NEURONS;

    if (!bool(pc.isTraining))
        return;


    CoopVec<float16_t, OUTPUT_NEURONS> outputGrad;

    result = exp(result);

    for (int i = 0; i < OUTPUT_NEURONS; i++)
    {
        //outputGrad[i] = (pow(texture.SampleLevel(randomUV, 0)[i], 1.f / 2.2) - pow(result[i], 1.f / 2.2)) * pow(result[i], 1.f / 2.2f);
        outputGrad[i] = (pow(texture.SampleLevel(randomUV, 0)[i], 1.f / 2.2) - pow(result[i], 1.f / 2.2)) * pow(result[i], 1.f / 2.2f);

        // dividing directly somehow rounds differently. Causes Inf in the batch summation
        outputGrad[i] *= 1.f / batch_size;
    }

    if (pc.isTraining == 1)
    {
        CoopVec<float16_t, HIDDEN_NEURONS> intermediateGrad = LinearMatMulAdd_Backwards(
            intermediateVecs[HIDDEN_LAYERS - 1], outputGrad, weights, weightAdjustment, buffOffset,
            buffOffset + pc.sizes[2], CoopVecMatrixLayout::TrainingOptimal, CoopVecComponentType::Float16);

        // for (int j = 0; j < HIDDEN_NEURONS; j++)
        // {
        //     float factor        = (intermediateVecs[HIDDEN_LAYERS - 1][j] < 0) ? 0.1f : 1.0f;
        //     intermediateGrad[j] = intermediateGrad[j] * factor;
        // }


        DifferentialPair<CoopVec<half, HIDDEN_NEURONS>> diffPairHidden = diffPair(intermediateVecs[HIDDEN_LAYERS - 1]);

        bwd_diff(LeakyReLU)(diffPairHidden, intermediateGrad);

        intermediateGrad = diffPairHidden.d;

        for (int i = HIDDEN_LAYERS - 2; i >= 0; i--)
        {
            buffOffset -= pc.sizes[1] + HIDDEN_NEURONS * 2;

            intermediateGrad = LinearMatMulAdd_Backwards(
                intermediateVecs[i], intermediateGrad, weights, weightAdjustment, buffOffset, buffOffset + pc.sizes[1],
                CoopVecMatrixLayout::TrainingOptimal, CoopVecComponentType::Float16);


            for (int j = 0; j < HIDDEN_NEURONS; j++)
            {
                float factor        = (intermediateVecs[i][j] < 0) ? 0.1f : 1.0f;
                intermediateGrad[j] = intermediateGrad[j] * factor;
            }
        }
        buffOffset = 0;
        CoopVec<half, INPUT_FEATURES> inputGrad =
            LinearMatMulAdd_Backwards(vec, intermediateGrad, weights, weightAdjustment, 0, pc.sizes[0],
                                      CoopVecMatrixLayout::TrainingOptimal, CoopVecComponentType::Float16);

        float2 dims;
        latentTexture.GetDimensions(dims.x, dims.y);

        float2 fractional = frac(randomUV * dims);
        int2   baseCoord  = floor(randomUV * dims);

        for (int x = 0; x < 2; x++)
            for (int y = 0; y < 2; y++)
            {

                // Lerp weight: if x=0, use (1-f); if x=1, use f.
                float weightX = (x == 0) ? (1.0 - fractional.x) : fractional.x;
                float weightY = (y == 0) ? (1.0 - fractional.y) : fractional.y;
                float factor  = weightX * weightY;

                // Ensure we don't wrap out of bounds and use Width (dims.x) for stride
                int coordX = clamp(baseCoord.x + x, 0, int(dims.x) - 1);
                int coordY = clamp(baseCoord.y + y, 0, int(dims.y) - 1);

                for (int i = 0; i < 4; i++)
                {
                    int index = (coordY * int(dims.x) + coordX) * 4 + i;

                    latentTrainingBuffer[index].add(inputGrad[INPUT_FEATURES - 4 + i] * factor);
                    // latentTrainingBuffer[index].add(inputGrad[INPUT_FEATURES - 4 + i] * factor * 100 *
                    // pc.learningRate);
                }
            }
    }
}
