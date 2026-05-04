#language slang 2025

//struct PushConstants
//{
//    DescriptorHandle<RWStructuredBuffer<int>> buffer;
//};

//[vk::push_constant]
//PushConstants pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID,
    uint  groupIndex : SV_GroupIndex
)
{
    DescriptorHandle<RWStructuredBuffer<int>> buffer = DescriptorHandle<RWStructuredBuffer<int>>(uint2(3024, 0));


    buffer[dispatchThreadID.x] = dispatchThreadID.x;
}
