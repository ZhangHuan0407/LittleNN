using System;
namespace LittleNN
{
    /// <summary>
    /// There are implement some of pytorch activation functions.
    /// </summary>
    [Serializable]
    public enum ActivationsFunctionType : int
    {
        Unknown = 0,
        /// <summary>
        /// Input layer don't have activations function, just padding function argument
        /// </summary>
        InputLayer,
        LeakyReLU,
        ReLU,
        Sigmoid,
        Softsign,
    }
}