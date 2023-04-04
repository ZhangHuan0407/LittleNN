using System;
using System.Collections.Generic;

namespace LittleNN
{
    /// <summary>
    /// Neuron link another Neuron which in connected layer
    /// </summary>
    public class Neuron
    {
        //public int IndexInLayer;
#pragma warning disable CS1591
        public float Bias;
        public float BiasDelta;
        public float Gradient;
        public float Value;
#pragma warning restore CS1591
        /// <summary>
        /// Each Synapse in InputSynapses is link to current Neuron
        /// </summary>
        public Synapse[] InputSynapses;
        /// <summary>
        /// Each Synapse in OutputSynapses is link from current Neuron
        /// </summary>
		public Synapse[] OutputSynapses;

        /// <summary>
        /// Create an empty instance
        /// </summary>
        protected Neuron()
        {
        }

        /// <summary>
        /// Create a Neuron with special Synapse length
        /// </summary>
        public static Neuron CreateNeuron(float bias, int inputLayerLength, int outputLayerLength)
        {
            Neuron neuron = new Neuron();
            neuron.Bias = bias;
            neuron.InputSynapses = new Synapse[inputLayerLength];
            neuron.OutputSynapses = new Synapse[outputLayerLength];
            return neuron;
        }
        /// <summary>
        /// Create a Neuron whitch in <see cref="NeuralNetwork.HiddenLayers"/> or <see cref="NeuralNetwork.OutputLayer"/>,
		/// new Neuron connect to each Neuron of inputLayer before return
        /// <para>InputSynapses:√</para>
        /// <para>OutputSynapses:√</para>
        /// <param name="index">new in Neuron sequence index in outputLayer layer</param>
        /// <param name="inputLayer">the layer connect with outputLayer</param>
        /// <param name="outputLayer">new Neuron will locate in outputLayer layer or null</param>
        /// </summary>
        public static Neuron CreateNeuronAndConnect(int index, NeuronLayer inputLayer, NeuronLayer outputLayer)
        {
            Neuron neuron = new Neuron();
            neuron.InputSynapses = new Synapse[inputLayer.NeuronsCount];
            if (outputLayer == default)
                neuron.OutputSynapses = Array.Empty<Synapse>();
            else
                neuron.OutputSynapses = new Synapse[outputLayer.NeuronsCount];
            for (int i = 0; i < inputLayer.NeuronsCount; i++)
            {
                Neuron inputNeuron = inputLayer.Neurons[i];
                var synapse = new Synapse(inputNeuron, neuron);
                inputNeuron.OutputSynapses[index] = synapse;
                neuron.InputSynapses[i] = synapse;
            }
            return neuron;
        }
        /// <summary>
        /// Create a Neuron whitch in <see cref="NeuralNetwork.InputLayer"/>
		/// <para>InputSynapses:x</para>
		/// <para>OutputSynapses:√</para>
        /// <param name="index">new in Neuron sequence index in outputLayer layer</param>
        /// <param name="outputLayer">new Neuron will locate in outputLayer layer</param>
        /// </summary>
		public static Neuron CreateInputLayerNeuron(int index, NeuronLayer outputLayer)
        {
            Neuron neuron = new Neuron();
            neuron.InputSynapses = Array.Empty<Synapse>();
            neuron.OutputSynapses = new Synapse[outputLayer.NeuronsCount];
            return neuron;
        }

#pragma warning disable CS1591
        public float CalculateValue(NeuronLayer layer)
        {
            Synapse[] inputSynapses = InputSynapses;
            float sum = 0f;
            for (int i = 0; i < inputSynapses.Length; i++)
            {
                Synapse synapse = InputSynapses[i];
                sum += synapse.Weight * synapse.InputNeuron.Value;
            }
            return Value = ActivationsFunctions.Output(layer.ActType, sum + Bias, layer.ActParameter);
        }

        public float CalculateError(float target)
        {
            return target - Value;
        }

        public float CalculateGradient(NeuronLayer layer, float? target)
        {
            float loss;
            if (target == null)
            {
                loss = 0f;
                Synapse[] outputSynapses = OutputSynapses;
                for (int i = 0; i < outputSynapses.Length; i++)
                {
                    Synapse synapse = OutputSynapses[i];
                    loss += synapse.OutputNeuron.Gradient * synapse.Weight;
                }
            }
            else
                loss = CalculateError(target.Value);
            return Gradient = loss * ActivationsFunctions.Derivative(layer.ActType, Value, layer.ActParameter);
        }

        public void UpdateWeights(float learnRate, float momentum)
        {
            var prevDelta = BiasDelta;
            BiasDelta = learnRate * Gradient;
            Bias += BiasDelta + momentum * prevDelta;

            Synapse[] inputSynapses = InputSynapses;
            for (int i = 0; i < inputSynapses.Length; i++)
            {
                Synapse synapse = InputSynapses[i];
                prevDelta = synapse.WeightDelta;
                synapse.WeightDelta = learnRate * Gradient * synapse.InputNeuron.Value;
                synapse.Weight += synapse.WeightDelta + momentum * prevDelta;
            }
        }
#pragma warning restore CS1591
    }
}