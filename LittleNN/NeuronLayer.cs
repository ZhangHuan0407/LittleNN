using System;

namespace LittleNN
{
    /// <summary>
    /// Contains Neuron in same layer
    /// </summary>
    public struct NeuronLayer
    {
        /// <summary>
        /// Neuron in same layer
        /// </summary>
        public Neuron[] Neurons;
        /// <summary>
        /// Neurons count
        /// </summary>
        public int NeuronsCount;

        /// <summary>
        /// Create a layer with special count
        /// </summary>
        public NeuronLayer(int neuronsCount)
        {
            NeuronsCount = neuronsCount;
            Neurons = new Neuron[neuronsCount];
        }
    }
}