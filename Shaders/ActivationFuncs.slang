#language 2026
module ActivationFuncs;


export public CoopVec<half, 16> FrequencyEncode(float2 uv)
{
    CoopVec<half, 16> frequencyEncoded;


    [ForceUnroll]
    for (int i = 1; i < 5; i++)
    {
        frequencyEncoded[0 + i * 4] = sin(uv[0] * pow(2, i) * acos(-1));
        frequencyEncoded[1 + i * 4] = sin(uv[1] * pow(2, i) * acos(-1));
        frequencyEncoded[2 + i * 4] = cos(uv[0] * pow(2, i) * acos(-1));
        frequencyEncoded[3 + i * 4] = cos(uv[1] * pow(2, i) * acos(-1));
    }

    return frequencyEncoded;
}

export public CoopVec<half, 16> SawtoothEncode(float2 uv)
{
    CoopVec<half, 16> frequencyEncoded;


    [ForceUnroll]
    for (int i = 1; i < 5; i++)
    {
        frequencyEncoded[0 + i * 4] = frac(uv.x * pow(2, i)) * 2 - 1;
        frequencyEncoded[1 + i * 4] = frac(uv.y * pow(2, i)) * 2 - 1;
        frequencyEncoded[2 + i * 4] = frac(uv.x * pow(2, i) + 0.25) * 2 - 1;
        frequencyEncoded[3 + i * 4] = frac(uv.y * pow(2, i) + 0.25) * 2 - 1;
    }

    return frequencyEncoded;
}


[TreatAsDifferentiable]
export public CoopVec<half, N> LeakyReLU<let N : int>(CoopVec<half, N> layer)
{
    // [ForceUnroll]
    // for (int i = 0; i < N; i++)
    // {
    //     layer[i] *= half(layer[i] > 0 ? 1 : 0.1);
    // }

    return max(layer, CoopVec<half, N>(half(0.))) + min(layer, CoopVec<half, N>(half(0.))) * half(0.1);
}

[BackwardDerivativeOf(LeakyReLU)]
export public void LeakyReLU_BackwardDiff<let N : int>(inout DifferentialPair<CoopVec<half, N>> p0,
                                                       CoopVec<half, N>.Differential            dOut)
{
    CoopVec<half, N>.Differential dIn = dOut; // copy first
    [ForceUnroll]
    for (int i = 0; i < N; i++)
        dIn[i] = dOut[i] * (p0.p[i] > half(0.) ? half(1.0) : half(0.1)); // mask by primal

    p0 = diffPair(p0.p, dIn);
}

export public void Softmax<let N : int>(inout CoopVec<half, N> layer)
{
    float16_t expSum  = 0;
    float16_t largest = layer[0];

    for (int i = 1; i < layer.getCount(); i++)
    {
        largest = (layer[i] > largest ? layer[i] : largest);
    }

    for (int i = 0; i < layer.getCount(); i++)
    {
        layer[i] -= largest;
    }
    layer = exp(layer);

    for (int i = 0; i < 10; i++)
    {
        expSum += layer[i];
    }

    for (int i = 0; i < 10; i++)
        layer[i] /= expSum;
}
