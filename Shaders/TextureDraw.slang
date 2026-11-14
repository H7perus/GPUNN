uint rotl(uint x, int k) {
    return (x << k) | (x >> (32 - k));
}

float pseudorandom(uint seed) {
    // Initialize state via splitmix32
    uint s = seed;

    uint s0, s1, s2, s3;
    s  = s + 0x9e3779b9u; s0 = s; s0 ^= s0 >> 16; s0 = s0 * 0x85ebca6bu; s0 ^= s0 >> 13; s0 = s0 * 0xc2b2ae35u; s0 ^= s0 >> 16;
    s  = s + 0x9e3779b9u; s1 = s; s1 ^= s1 >> 16; s1 = s1 * 0x85ebca6bu; s1 ^= s1 >> 13; s1 = s1 * 0xc2b2ae35u; s1 ^= s1 >> 16;
    s  = s + 0x9e3779b9u; s2 = s; s2 ^= s2 >> 16; s2 = s2 * 0x85ebca6bu; s2 ^= s2 >> 13; s2 = s2 * 0xc2b2ae35u; s2 ^= s2 >> 16;
    s  = s + 0x9e3779b9u; s3 = s; s3 ^= s3 >> 16; s3 = s3 * 0x85ebca6bu; s3 ^= s3 >> 13; s3 = s3 * 0xc2b2ae35u; s3 ^= s3 >> 16;

    // xoshiro128** step
    uint result = rotl(s1 * 5u, 7) * 9u;
    uint t      = s1 << 9;
    s2 ^= s0;
    s3 ^= s1;
    s1 ^= s2;
    s0 ^= s3;
    s2 ^= t;
    s3 = rotl(s3, 11);

    // Map to [0, 1)
    return float(result) * (1.0 / 4294967296.0);
}





struct VSOutput {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD;
};

struct VSInput {
    float3 pos   : POSITION;  // location 0
};

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    output.pos = float4(input.pos * float3(0.5, 1, 1) - float3(0.5, 0, 0), 1.0);
    output.uv  = input.pos.xy * 0.5 + 0.5;
    return output;
}

struct PushConstants {
    int texIndex;
}
[[vk::push_constant]]
PushConstants pc;

// fragment shader
//[require(spvDescriptorHeapEXT)]
[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target {

    DescriptorHandle<Sampler2D> texture = DescriptorHandle<Sampler2D>(uint2(3026, 16));

    //float4 returnval = pseudorandom(input.uv.x * 600 * 600 + input.uv.y * 600);

    return texture.Sample(input.uv);
}