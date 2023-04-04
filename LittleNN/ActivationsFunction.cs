using System;

namespace LittleNN
{
    /// <summary>
    /// https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/#non-linear-activations-source
    /// copy from website
    /// </summary>
    public static class ActivationsFunctions
    {
        /// <summary>
        /// Extend more custom <see cref="ActivationsFunctionType"/>, and extend <see cref="Output"/> by add delegate to <see cref="OutputExtension_Handle"/>
        /// </summary>
        public delegate float ActivationsFunction(ActivationsFunctionType type, float x, float parameter);
        /// <summary>
        /// Extend more custom <see cref="ActivationsFunctionType"/>, and extend <see cref="Output"/> by add delegate to <see cref="OutputExtension_Handle"/>
        /// </summary>
        public static event ActivationsFunction OutputExtension_Handle;
        /// <summary>
        /// Calculate activation function result
        /// <para>If type is not build-in, will invoke <see cref="OutputExtension_Handle"/> to get activation</para>
        /// </summary>
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

        /// <summary>
        /// Extend more custom <see cref="ActivationsFunctionType"/>, and extend <see cref="Output"/> by add delegate to <see cref="DerivativeExtension_Handle"/>
        /// </summary>
        public delegate float DerivativeFunction(ActivationsFunctionType type, float x, float parameter);
        /// <summary>
        /// Extend more custom <see cref="ActivationsFunctionType"/>, and extend <see cref="Output"/> by add delegate to <see cref="DerivativeExtension_Handle"/>
        /// </summary>
        public static event DerivativeFunction DerivativeExtension_Handle;
        /// <summary>
        /// Calculate derivative function result
        /// <para>If type is not build-in, will invoke <see cref="DerivativeExtension_Handle"/> to get derivative</para>
        /// </summary>
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