using System;
namespace LittleNN
{
    /// <summary>
    /// There are implement some of pytorch activation functions.
    /// </summary>
    public enum ActivationsFunctionType : int
    {
        /// <summary>
        /// Default value 0 convert to <see cref="Unknown"/>,
        /// forget to set type value?
        /// </summary>
        Unknown = 0,
        /// <summary>
        /// Input layer don't have activations function, just padding function argument
        /// </summary>
        InputLayer,
        /// <summary>
        /// x > 0 ? x : (0.02 * x)
        /// </summary>
        LeakyReLU,
        /// <summary>
        /// max(x, 0)
        /// </summary>
        ReLU,
        /// <summary>
        /// 1f / (1f + MathF.Exp(-x))
        /// </summary>
        Sigmoid,
        /// <summary>
        /// x / (1f + |x|);
        /// </summary>
        Softsign,
    }
}