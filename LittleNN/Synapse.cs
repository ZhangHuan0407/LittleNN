using System;

namespace LittleNN
{
    /// <summary>
    /// Link two Neuron which in connected layer
    /// </summary>
	public class Synapse
    {
        /// <summary>
        /// Synapse link from InputNeuron to OutputNeuron
        /// </summary>
        public Neuron InputNeuron;
        /// <summary>
        /// Synapse link from InputNeuron to OutputNeuron
        /// </summary>
		public Neuron OutputNeuron;
        /// <summary>
        /// Link weight from InputNeuron.Value to OutputNeuron.Value
        /// </summary>
        public float Weight;
#pragma warning disable CS1591
        public float WeightDelta;
#pragma warning restore CS1591

        /// <summary>
        /// Create an empty instance
        /// </summary>
        public Synapse() { }

        /// <summary>
        /// Create an Synapse link from inputNeuron to outputNeuron
        /// </summary>
        public Synapse(Neuron inputNeuron, Neuron outputNeuron)
        {
            InputNeuron = inputNeuron ?? throw new ArgumentNullException(nameof(inputNeuron));
            OutputNeuron = outputNeuron ?? throw new ArgumentNullException(nameof(outputNeuron));
            Weight = NeuralNetwork.GetRandom();
        }
    }
}