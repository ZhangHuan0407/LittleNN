using System;
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
        /// Create a new neural network with special layer size and parameters
        /// </summary>
		public NeuralNetwork(int inputSize, int[] hiddenSizes, int outputSize, float learnRate = 0.02f, float momentum = 0.75f)
        {
            if (learnRate < 0f || learnRate >= 1.000001f)
                throw new ArgumentException($"{learnRate} out of [0,1]");
            LearnRate = learnRate;
            if (momentum < 0f || momentum >= 1.000001f)
                throw new ArgumentException($"{momentum} out of [0,1]");
            if (hiddenSizes.Length == 0 || hiddenSizes.Contains(0))
                throw new ArgumentException(nameof(hiddenSizes));
            Momentum = momentum;
            InputLayer = new NeuronLayer(inputSize);
            HiddenLayers = new NeuronLayer[hiddenSizes.Length];
            for (int i = 0; i < hiddenSizes.Length; i++)
                HiddenLayers[i] = new NeuronLayer(hiddenSizes[i]);
            OutputLayer = new NeuronLayer(outputSize);

            for (var i = 0; i < inputSize; i++)
                InputLayer.Neurons[i] = Neuron.CreateInputNeuron(i, HiddenLayers[0]);

            for (var i = 0; i < hiddenSizes.Length; i++)
            {
                int hiddenSize = hiddenSizes[i];
                NeuronLayer hiddenLayer = HiddenLayers[i];
                NeuronLayer inputNeuronsLayer = i > 0 ? HiddenLayers[i - 1] : InputLayer;
                NeuronLayer outputNeuronsLayer = i < hiddenSizes.Length - 1 ? HiddenLayers[i + 1] : OutputLayer;
                for (var j = 0; j < hiddenSize; j++)
                    hiddenLayer.Neurons[j] = Neuron.CreateHiddenNeuron(j, inputNeuronsLayer, outputNeuronsLayer);
                HiddenLayers[i] = hiddenLayer;
            }

            NeuronLayer lastNeuronsLayer = HiddenLayers[HiddenLayers.Length - 1];
            for (var i = 0; i < outputSize; i++)
                OutputLayer.Neurons[i] = Neuron.CreateOutputNeuron(i, lastNeuronsLayer);
        }

        private void ForwardPropagate(float[] inputs)
        {
            for (int index = 0; index < InputLayer.NeuronsCount; index++)
                InputLayer.Neurons[index].Value = inputs[index];
            for (int layerIndex = 0; layerIndex < HiddenLayers.Length; layerIndex++)
            {
                NeuronLayer hiddenLayer = HiddenLayers[layerIndex];
                for (int nIndex = 0; nIndex < hiddenLayer.NeuronsCount; nIndex++)
                    hiddenLayer.Neurons[nIndex].CalculateValue();
            }
            for (int index = 0; index < OutputLayer.NeuronsCount; index++)
                OutputLayer.Neurons[index].CalculateValue();
        }

        private void BackPropagate(float[] targets)
        {
            for (int index = 0; index < OutputLayer.NeuronsCount; index++)
                OutputLayer.Neurons[index].CalculateGradient(targets[index]);
            for (int layerIndex = HiddenLayers.Length - 1; layerIndex >= 0; layerIndex--)
            {
                NeuronLayer hiddenLayer = HiddenLayers[layerIndex];
                for (int nIndex = 0; nIndex < hiddenLayer.NeuronsCount; nIndex++)
                    hiddenLayer.Neurons[nIndex].CalculateGradient();
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