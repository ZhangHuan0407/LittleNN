using System;
using System.Collections.Generic;

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
        /// <summary>
        /// The parameter of activation function, example: <see cref="ActivationsFunctionType.LeakyReLU"/> use 0.02f
        /// </summary>
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

#pragma warning disable CS1591
        public override bool Equals(object obj)
        {
            return obj is NeuronLayer layer &&
                   NeuronsCount == layer.NeuronsCount &&
                   EqualityComparer<Neuron[]>.Default.Equals(Neurons, layer.Neurons) &&
                   ActType == layer.ActType &&
                   ActParameter == layer.ActParameter;
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(NeuronsCount, Neurons, ActType, ActParameter);
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
#pragma warning restore CS1591
    }
}