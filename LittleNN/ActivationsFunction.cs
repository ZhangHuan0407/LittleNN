using System;

namespace LittleNN
{
    /// <summary>
    /// https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/#non-linear-activations-source
    /// copy from website
    /// </summary>
    public static class ActivationsFunctions
    {
        public delegate float ActivationsFunction(ActivationsFunctionType type, float x, float parameter);
        public static event ActivationsFunction OutputExtension_Handle;
        public static float Output(ActivationsFunctionType type, float x, float parameter)
        {
            switch (type)
            {
                case ActivationsFunctionType.LeakyReLU:
                    return LeakyRelu(x, parameter);
                case ActivationsFunctionType.ReLU:
                    return ReLU(x);
                case ActivationsFunctionType.Sigmoid:
                    return Sigmoid(x);
                case ActivationsFunctionType.Softsign:
                    return Softsign(x);
                case ActivationsFunctionType.Unknown:
                default:
                    float? value = OutputExtension_Handle?.Invoke(type, x, parameter);
                    if (value is null)
                        throw new Exception($"Unknown ActivationsFunction type: {type}");
                    return value.Value;
            }
            static float LeakyRelu(float x, float negativeSlope)
            {
                if (x >= 0f)
                    return x;
                else
                    return x * negativeSlope;
            }
            static float ReLU(float x)
            {
                if (x >= 0f)
                    return x;
                else
                    return 0f;
            }
            static float Sigmoid(float x)
            {
                if (x < -45f)
                    return 0f;
                else if (x > 45f)
                    return 1f;
                else
                    return 1f / (1f + MathF.Exp(-x));
            }
            static float Softsign(float x)
            {
                float abs = x > 0f ? x : -x;
                return x / (1f + abs);
            }
        }

        public delegate float DerivativeFunction(ActivationsFunctionType type, float x, float parameter);
        public static event DerivativeFunction DerivativeExtension_Handle;
        public static float Derivative(ActivationsFunctionType type, float x, float parameter)
        {
            switch (type)
            {
                case ActivationsFunctionType.LeakyReLU:
                    return LeakyRelu(x, parameter);
                case ActivationsFunctionType.ReLU:
                    return ReLU(x);
                case ActivationsFunctionType.Sigmoid:
                    return Sigmoid(x);
                case ActivationsFunctionType.Softsign:
                    return Softsign(x);
                case ActivationsFunctionType.Unknown:
                default:
                    float? value = DerivativeExtension_Handle?.Invoke(type, x, parameter);
                    if (value is null)
                        throw new Exception($"Unknown DerivativeFunction type: {type}");
                    return value.Value;
            }
            static float LeakyRelu(float x, float negativeSlope)
            {
                if (x >= 0f)
                    return 1f;
                else
                    return negativeSlope;
            }
            static float ReLU(float x)
            {
                if (x >= 0f)
                    return 1f;
                else
                    return 0f;
            }
            static float Sigmoid(float x) => x * (1 - x);
            static float Softsign(float x)
            {
                float abs = x > 0f ? x : -x;
                x = (1f + abs);
                return 1f / (x * x);
            }
        }
    }
}