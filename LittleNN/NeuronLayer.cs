using System;

namespace LittleNN
{
    /// <summary>
    /// Contains Neuron in same layer
    /// </summary>
    public struct NeuronLayer
    {
        /// <summary>
        /// Neurons count
        /// </summary>
        public int NeuronsCount;
        /// <summary>
        /// Neuron in same layer
        /// </summary>
        public Neuron[] Neurons;
        /// <summary>
        /// Activations Function Type of Neurons in this layer
        /// </summary>
        public ActivationsFunctionType ActType;
        public float ActParameter;

        /// <summary>
        /// Create a layer with special count
        /// </summary>
        public NeuronLayer(int neuronsCount, ActivationsFunctionType type, float? parameter)
        {
            NeuronsCount = neuronsCount;
            Neurons = new Neuron[neuronsCount];
            ActType = type;
            if (parameter is null)
            {
                parameter = 0f;
                if (type == ActivationsFunctionType.LeakyReLU)
                    parameter = 0.02f;
            }
            ActParameter = parameter.Value;
        }

        public static bool operator ==(NeuronLayer left, NeuronLayer right)
        {
            return left.Neurons == right.Neurons &&
                   left.ActType == right.ActType &&
                   left.ActParameter == right.ActParameter;
        }
        public static bool operator !=(NeuronLayer left, NeuronLayer right)
        {
            return left.Neurons != right.Neurons ||
                   left.ActType != right.ActType ||
                   left.ActParameter != right.ActParameter;
        }
    }
}