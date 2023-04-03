using System;
using System.Collections.Generic;
using System.Linq;

namespace LittleNN
{
    public partial class NeuralNetwork
    {
        /// <summary>
        /// Control each learning effect.
		/// <para>Too high a learning rate <see cref="NeuralNetwork"/> will not converge</para>
		/// <para>Too samll a learning rate <see cref="NeuralNetwork"/> will converge slowly</para>
        /// </summary>
        public float LearnRate;
        /// <summary>
        /// Control each learning Momentum.
		/// <para>Too high a momentum <see cref="NeuralNetwork"/> will not converge</para>
		/// <para>Too samll a momentum <see cref="NeuralNetwork"/> will local overfitting</para>
        /// </summary>
		public float Momentum;
        /// <summary>
        /// The input layer of <see cref="NeuralNetwork"/>
        /// </summary>
		public NeuronLayer InputLayer;
        /// <summary>
        /// The hidden layer of <see cref="NeuralNetwork"/>
        /// </summary>
		public NeuronLayer[] HiddenLayers;
        /// <summary>
        /// The out layer of <see cref="NeuralNetwork"/>
        /// </summary>
        public NeuronLayer OutputLayer;

        /// <summary>
        /// Create an empty instance
        /// </summary>
        public NeuralNetwork()
        {
            LearnRate = 0;
            Momentum = 0;
            InputLayer = default;
            HiddenLayers = Array.Empty<NeuronLayer>();
            OutputLayer = default;
        }

        /// <summary>
        /// Create a new neural network with special layer size and default parameters.
        /// <para><see cref="NeuronLayer.ActType"/>all synapse choose Sigmoid as activation function</para>
        /// </summary>
		public NeuralNetwork(int inputSize, int[] hiddenSizes, int outputSize, float learnRate = 0.02f, float momentum = 0.75f)
        {
            List<Sequential> sequential = Sequential.CreateNew();
            sequential.Add(Sequential.Neural("input layer", inputSize));
            for (int i = 0; i < hiddenSizes.Length; i++)
                sequential.Add(Sequential.Neural("hidden layer", hiddenSizes[i]));
            sequential.Add(Sequential.Neural("output layer", outputSize));
            SetNeuralNetwork(sequential, learnRate, momentum);
        }
        public NeuralNetwork(List<Sequential> sequential, float learnRate, float momentum) =>
            SetNeuralNetwork(sequential, learnRate, momentum);

        private void SetNeuralNetwork(List<Sequential> sequential, float learnRate, float momentum)
        {
            if (learnRate < 0f || learnRate >= 1.000001f)
                throw new ArgumentException($"{learnRate} out of [0,1]");
            LearnRate = learnRate;
            if (momentum < 0f || momentum >= 1.000001f)
                throw new ArgumentException($"{momentum} out of [0,1]");
            Momentum = momentum;

            int layerCount = sequential.LayerCount();
            if (layerCount < 3)
                throw new ArgumentException($"sequential layers less than 3");
            int[] allLayerSize = new int[layerCount];
            ActivationsFunctionType[] actTypes = new ActivationsFunctionType[layerCount - 1];
            float?[] actParameters = new float?[layerCount - 1];
            int layerIndex = 0;
            for (int i = 0; i < sequential.Count; i++)
            {
                Sequential seq = sequential[i];
                if (seq.ParameterType == Sequential.SequentialParameter.LayerNeuralCount)
                {
                    allLayerSize[layerIndex] = seq.NeuralCount;
                    layerIndex++;
                }
                else if (seq.ParameterType == Sequential.SequentialParameter.ActivationFunction)
                {
                    actTypes[layerIndex - 1] = seq.ActType;
                    actParameters[layerIndex - 1] = seq.ActParameter;
                }
            }
            for (int i = 0; i < actTypes.Length; i++)
            {
                if (actTypes[i] == ActivationsFunctionType.Unknown)
                    actTypes[i] = ActivationsFunctionType.Sigmoid;
            }

            InputLayer = new NeuronLayer(allLayerSize[0], ActivationsFunctionType.InputLayer, null);
            HiddenLayers = new NeuronLayer[layerCount - 2];
            for (int i = 0; i < layerCount - 2; i++)
                HiddenLayers[i] = new NeuronLayer(allLayerSize[i + 1], actTypes[i], actParameters[i]);
            OutputLayer = new NeuronLayer(allLayerSize[layerCount - 1], actTypes[layerCount - 2], actParameters[layerCount - 2]);

            for (var i = 0; i < allLayerSize[0]; i++)
                InputLayer.Neurons[i] = Neuron.CreateInputLayerNeuron(i, HiddenLayers[0]);

            for (var i = 0; i < layerCount - 2; i++)
            {
                int hiddenSize = allLayerSize[i + 1];
                NeuronLayer hiddenLayer = HiddenLayers[i];
                NeuronLayer inputNeuronsLayer = i > 0 ? HiddenLayers[i - 1] : InputLayer;
                NeuronLayer outputNeuronsLayer = i < layerCount - 3 ? HiddenLayers[i + 1] : OutputLayer;
                for (var j = 0; j < hiddenSize; j++)
                    hiddenLayer.Neurons[j] = Neuron.CreateNeuronAndConnect(j, inputNeuronsLayer, outputNeuronsLayer);
                HiddenLayers[i] = hiddenLayer;
            }

            NeuronLayer lastNeuronsLayer = HiddenLayers[HiddenLayers.Length - 1];
            for (var i = 0; i < allLayerSize[layerCount - 1]; i++)
                OutputLayer.Neurons[i] = Neuron.CreateNeuronAndConnect(i, lastNeuronsLayer, default);
        }

        private void ForwardPropagate(float[] inputs)
        {
            for (int index = 0; index < InputLayer.NeuronsCount; index++)
                InputLayer.Neurons[index].Value = inputs[index];
            for (int layerIndex = 0; layerIndex < HiddenLayers.Length; layerIndex++)
            {
                NeuronLayer hiddenLayer = HiddenLayers[layerIndex];
                for (int nIndex = 0; nIndex < hiddenLayer.NeuronsCount; nIndex++)
                    hiddenLayer.Neurons[nIndex].CalculateValue(hiddenLayer);
            }
            for (int index = 0; index < OutputLayer.NeuronsCount; index++)
                OutputLayer.Neurons[index].CalculateValue(OutputLayer);
        }

        private void BackPropagate(float[] targets)
        {
            for (int index = 0; index < OutputLayer.NeuronsCount; index++)
                OutputLayer.Neurons[index].CalculateGradient(OutputLayer, targets[index]);
            for (int layerIndex = HiddenLayers.Length - 1; layerIndex >= 0; layerIndex--)
            {
                NeuronLayer hiddenLayer = HiddenLayers[layerIndex];
                for (int nIndex = 0; nIndex < hiddenLayer.NeuronsCount; nIndex++)
                    hiddenLayer.Neurons[nIndex].CalculateGradient(hiddenLayer);
            }
            for (int layerIndex = HiddenLayers.Length - 1; layerIndex >= 0; layerIndex--)
            {
                NeuronLayer hiddenLayer = HiddenLayers[layerIndex];
                for (int nIndex = 0; nIndex < hiddenLayer.NeuronsCount; nIndex++)
                    hiddenLayer.Neurons[nIndex].UpdateWeights(LearnRate, Momentum);
            }
            for (int index = 0; index < OutputLayer.NeuronsCount; index++)
                OutputLayer.Neurons[index].UpdateWeights(LearnRate, Momentum);
        }

        private float[] CopyEval()
        {
            float[] eval = new float[OutputLayer.NeuronsCount];
            for (int i = 0; i < OutputLayer.NeuronsCount; i++)
                eval[i] = OutputLayer.Neurons[i].Value;
            return eval;
        }
    }
}