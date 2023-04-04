using System;
using System.Buffers;
using System.IO;

namespace LittleNN
{
    /// <summary>
    /// Copy <see cref="NeuralNetwork"/> parameter and serialize/deserialize
    /// </summary>
    public class NeuralNetworkModel
    {
        private const int CurrentVersionHeadSize = 20;
        private const int CurrentVersion = 2;

        /// <summary>
        /// bin data head area size
        /// </summary>
        public int HeadSize;
        /// <summary>
        /// You can not use model in accross version
        /// </summary>
        public int Version;
        /// <summary>
        /// <see cref="NeuralNetwork.LearnRate"/> backup
        /// </summary>
        public float LearnRate;
        /// <summary>
        /// <see cref="NeuralNetwork.Momentum"/> backup
        /// </summary>
        public float Momentum;
        /// <summary>
        /// <see cref="NeuralNetwork"/> neuron count of each layer
        /// </summary>
        public int[] LayerRank;
        /// <summary>
        /// [HiddenLayer 1~HiddenLayer n OutputLayer][NeuronIndex]' ActivationsFunctionType
        /// </summary>
        public int[] ActivationsFunctionTypes;
        /// <summary>
        /// [HiddenLayer 1~HiddenLayer n OutputLayer][NeuronIndex]' ActivationsFunctionType
        /// </summary>
        public float[] ActivationsFunctionParameters;
        /// <summary>
        /// [HiddenLayer 1~HiddenLayer n OutputLayer][NeuronIndex]' Bias
        /// <para>
        /// <see cref="NeuronBias"/>.GetLength(0) is equal <see cref="LayerRank"/>.Length - 1
        /// <see cref="NeuronBias"/>[i].Length is equal <see cref="LayerRank"/>[i]
        /// </para>
        /// </summary>
        public float[][] NeuronBias;
        /// <summary>
        /// [InputLayer HiddenLayer 1~HiddenLayer n][neuronLayer+1 link to neuronLayer]' weight
        /// <para>
        /// <see cref="SynapseWeight"/>.GetLength(0) is equal <see cref="LayerRank"/>.Length - 1
        /// <see cref="NeuronBias"/>[i].Length is equal <see cref="LayerRank"/>[i] * <see cref="LayerRank"/>[i + 1]
        /// </para>
        /// </summary>
        public float[][] SynapseWeight;

        /// <summary>
        /// Create a empty model
        /// </summary>
        public NeuralNetworkModel()
        {
        }

        /// <summary>
        /// Copy <paramref name="neuralNetwork"/> value to <see cref="NeuralNetworkModel"/>,
        /// all of NeuralNetwork forward parameter create a backup.
        /// </summary>
        public void CopyFrom(NeuralNetwork neuralNetwork)
        {
            Version = CurrentVersion;
            LearnRate = neuralNetwork.LearnRate;
            Momentum = neuralNetwork.Momentum;
            HeadSize = CurrentVersionHeadSize;

            int layerCount = neuralNetwork.HiddenLayers.Length + 2;
            LayerRank = new int[layerCount];
            ActivationsFunctionTypes = new int[layerCount - 1];
            ActivationsFunctionParameters = new float[layerCount - 1];
            NeuronBias = new float[layerCount - 1][];
            SynapseWeight = new float[layerCount - 1][];

            int layerIndex = 0;
            NeuronLayer neuronLayer = neuralNetwork.InputLayer;
            LayerRank[layerIndex] = neuronLayer.NeuronsCount;
            SynapseWeight[layerIndex] = CopyLayerSynapseToArray(neuronLayer);
            layerIndex++;

            for (int i = 0; i < neuralNetwork.HiddenLayers.Length; i++)
            {
                neuronLayer = neuralNetwork.HiddenLayers[i];
                LayerRank[layerIndex] = neuronLayer.NeuronsCount;
                ActivationsFunctionTypes[layerIndex - 1] = (int)neuronLayer.ActType;
                ActivationsFunctionParameters[layerIndex - 1] = neuronLayer.ActParameter;
                NeuronBias[layerIndex - 1] = CopyLayerNeuronToArray(neuronLayer);
                SynapseWeight[layerIndex] = CopyLayerSynapseToArray(neuronLayer);
                layerIndex++;
            }

            neuronLayer = neuralNetwork.OutputLayer;
            LayerRank[layerIndex] = neuronLayer.NeuronsCount;
            ActivationsFunctionTypes[layerIndex - 1] = (int)neuronLayer.ActType;
            ActivationsFunctionParameters[layerIndex - 1] = neuronLayer.ActParameter;
            NeuronBias[layerIndex - 1] = CopyLayerNeuronToArray(neuronLayer);
        }
        private float[] CopyLayerNeuronToArray(NeuronLayer neuronLayer)
        {
            float[] bias = new float[neuronLayer.NeuronsCount];
            for (int i = 0; i < neuronLayer.NeuronsCount; i++)
            {
                bias[i] = neuronLayer.Neurons[i].Bias;
            }
            return bias;
        }
        /// <summary>
        /// return: float[neuronLayer+1 link to neuronLayer]'s weight
        /// </summary>
        private float[] CopyLayerSynapseToArray(NeuronLayer neuronLayer)
        {
            int synapseCountPreNeuron = neuronLayer.Neurons[0].OutputSynapses.Length;
            float[] weight = new float[synapseCountPreNeuron * neuronLayer.NeuronsCount];
            for (int i = 0; i < neuronLayer.NeuronsCount; i++)
            {
                Synapse[] synapses = neuronLayer.Neurons[i].OutputSynapses;
                for (int j = 0; j < synapseCountPreNeuron; j++)
                {
                    // todo there is a foreach performance mistake, point order is not constant
                    int point = j * neuronLayer.NeuronsCount + i;
                    weight[point] = synapses[j].Weight;
                }
            }
            return weight;
        }

        /// <summary>
        /// Use <see cref="NeuralNetworkModel"/> value override <paramref name="neuralNetwork"/> content,
        /// all of NeuralNetwork forward parameter will return to model's backup.
        /// </summary>
        public void Override(NeuralNetwork neuralNetwork)
        {
            neuralNetwork.LearnRate = LearnRate;
            neuralNetwork.Momentum = Momentum;
            neuralNetwork.InputLayer = new NeuronLayer(LayerRank[0], ActivationsFunctionType.InputLayer, 0f);
            neuralNetwork.HiddenLayers = new NeuronLayer[LayerRank.Length - 2];
            neuralNetwork.OutputLayer = new NeuronLayer(neuronsCount: LayerRank[LayerRank.Length - 1],
                                                        type: (ActivationsFunctionType)ActivationsFunctionTypes[LayerRank.Length - 2],
                                                        parameter: ActivationsFunctionParameters[LayerRank.Length - 2]);
            NeuronLayer neuronLayer = neuralNetwork.InputLayer;
            float[] neuronBias;
            for (int i = 0; i < neuronLayer.NeuronsCount; i++)
            {
                neuronLayer.Neurons[i] = Neuron.CreateNeuron(0f, 0, LayerRank[1]);
            }
            for (int i = 0; i < neuralNetwork.HiddenLayers.Length; i++)
            {
                neuronLayer = neuralNetwork.HiddenLayers[i] = new NeuronLayer(neuronsCount: LayerRank[i + 1],
                                                                              type: (ActivationsFunctionType)ActivationsFunctionTypes[i],
                                                                              parameter: ActivationsFunctionParameters[i]);
                neuronBias = NeuronBias[i];
                int inputLayerRank = LayerRank[i];
                int outputLayerRank = LayerRank[i + 2];
                for (int j = 0; j < neuronLayer.NeuronsCount; j++)
                    neuronLayer.Neurons[j] = Neuron.CreateNeuron(neuronBias[j], inputLayerRank, outputLayerRank);
            }
            neuronLayer = neuralNetwork.OutputLayer;
            neuronBias = NeuronBias[NeuronBias.Length - 1];
            for (int i = 0; i < neuronLayer.NeuronsCount; i++)
                neuronLayer.Neurons[i] = Neuron.CreateNeuron(neuronBias[i], LayerRank[LayerRank.Length - 2], 0);

            LinkLayerSynapse(SynapseWeight[0], neuralNetwork.InputLayer, neuralNetwork.HiddenLayers[0]);
            for (int i = 0; i < neuralNetwork.HiddenLayers.Length; i++)
            {
                NeuronLayer outputLayer = i < neuralNetwork.HiddenLayers.Length - 1 ? neuralNetwork.HiddenLayers[i + 1] : neuralNetwork.OutputLayer;
                LinkLayerSynapse(SynapseWeight[i + 1], neuralNetwork.HiddenLayers[i], outputLayer);
            }
        }
        private void LinkLayerSynapse(float[] weight, NeuronLayer aLayer, NeuronLayer bLayer)
        {
            int synapseCountPreNeuron = bLayer.NeuronsCount;
            for (int i = 0; i < aLayer.NeuronsCount; i++)
            {
                Neuron aNeuron = aLayer.Neurons[i];
                for (int j = 0; j < synapseCountPreNeuron; j++)
                {
                    Neuron bnNeuron = bLayer.Neurons[j];
                    Synapse synapse = new Synapse(aNeuron, bnNeuron);
                    int point = j * aLayer.NeuronsCount + i;
                    synapse.Weight = weight[point];
                    aNeuron.OutputSynapses[j] = synapse;
                    bnNeuron.InputSynapses[i] = synapse;
                }
            }
        }

        /// <summary>
        /// Direct use model forward calculate without <see cref="NeuralNetwork"/>.
        /// You can't train neural network by this way.
        /// </summary>
        /// <param name="input">input value of neural network</param>
        /// <returns>forward calculate result</returns>
        public float[] QuickForward(float[] input)
        {
            if (input == null || input.Length != LayerRank[0])
                throw new ArgumentException(nameof(input));
            if (LayerRank == null || LayerRank.Length == 0)
                throw new Exception(nameof(LayerRank));

            float[] aLayerValue = null;
            float[] bLayerValue = input;

            // translate value from [calculateTimes]layer to [calculateTimes+1]layer
            for (int calculateTimes = 0; calculateTimes < LayerRank.Length - 1; calculateTimes++)
            {
                if (aLayerValue != null && aLayerValue != input)
                    ArrayPool<float>.Shared.Return(aLayerValue);
                aLayerValue = bLayerValue;
                int aLayerNeuronCount = LayerRank[calculateTimes];
                int bLayerNeuronCount = LayerRank[calculateTimes + 1];
                float[] synapseWeight = SynapseWeight[calculateTimes];
                float[] neuronBias = NeuronBias[calculateTimes];
                bLayerValue = ArrayPool<float>.Shared.Rent(bLayerNeuronCount);
                int synapsePoint = 0;
                ActivationsFunctionType actType = (ActivationsFunctionType)ActivationsFunctionTypes[calculateTimes];
                float actParameter = ActivationsFunctionParameters[calculateTimes];
                for (int nIndex = 0; nIndex < bLayerNeuronCount; nIndex++)
                {
                    float sum = 0f;
                    for (int i = 0; i < aLayerNeuronCount; i++)
                        sum += synapseWeight[synapsePoint++] * aLayerValue[i];
                    bLayerValue[nIndex] = ActivationsFunctions.Output(actType, sum + neuronBias[nIndex], actParameter);
                }
            }
            ArrayPool<float>.Shared.Return(bLayerValue);
            float[] eval = new float[LayerRank[LayerRank.Length - 1]];
            Array.Copy(bLayerValue, eval, eval.Length);
            return eval;
        }

        /// <summary>
        /// Wrtite model data with bin format
        /// </summary>
        public void Write(Stream stream)
        {
            if (stream is null || !stream.CanWrite)
                throw new ArgumentException(nameof(stream));
            long headStartPosition = stream.Position;
            WriteInt(stream, HeadSize);
            WriteInt(stream, Version);
            WriteFloat(stream, LearnRate);
            WriteFloat(stream, Momentum);
            if (headStartPosition + HeadSize - stream.Position != 0)
            {
                byte[] padding = new byte[headStartPosition + HeadSize - stream.Position];
                stream.Write(padding, 0, padding.Length);
            }
            // LayerRank
            int layerRankLength = LayerRank.Length;
            WriteInt(stream, layerRankLength);
            for (int i = 0; i < layerRankLength; i++)
                WriteInt(stream, LayerRank[i]);
            // ActivationsFunctionTypes
            for (int i = 0; i < ActivationsFunctionTypes.Length; i++)
                WriteInt(stream, ActivationsFunctionTypes[i]);
            // ActivationsFunctionParameters
            WriteFloatArray(stream, ActivationsFunctionParameters);
            // NeuronBias
            for (int i = 0; i < layerRankLength - 1; i++)
                WriteFloatArray(stream, NeuronBias[i]);
            // SynapseWeight
            for (int i = 0; i < layerRankLength - 1; i++)
                WriteFloatArray(stream, SynapseWeight[i]);
        }
        /// <summary>
        /// Read model data with bin format
        /// </summary>
        public void Read(Stream stream)
        {
            if (stream is null || !stream.CanRead)
                throw new ArgumentException(nameof(stream));
            byte[] buffer = new byte[4];
            long headStartPosition = stream.Position;
            HeadSize = ReadInt(stream, buffer);
            Version = ReadInt(stream, buffer);
            LearnRate = ReadFloat(stream, buffer);
            Momentum = ReadFloat(stream, buffer);
            stream.Position = headStartPosition + HeadSize;
            // LayerRank
            int layerRankLength = ReadInt(stream, buffer);
            LayerRank = new int[layerRankLength];
            for (int i = 0; i < layerRankLength; i++)
                LayerRank[i] = ReadInt(stream, buffer);
            // ActivationsFunctionTypes
            ActivationsFunctionTypes = new int[layerRankLength - 1];
            for (int i = 0; i < layerRankLength - 1; i++)
                ActivationsFunctionTypes[i] = ReadInt(stream, buffer);
            // ActivationsFunctionParameters
            ActivationsFunctionParameters = ReadFloatArray(stream, layerRankLength - 1);
            // NeuronBias
            NeuronBias = new float[layerRankLength - 1][];
            for (int i = 0; i < layerRankLength - 1; i++)
                NeuronBias[i] = ReadFloatArray(stream, LayerRank[i + 1]);
            // SynapseWeight
            SynapseWeight = new float[layerRankLength - 1][];
            for (int i = 0; i < layerRankLength - 1; i++)
                SynapseWeight[i] = ReadFloatArray(stream, LayerRank[i] * LayerRank[i + 1]);
        }

        private void WriteFloatArray(Stream stream, float[] array)
        {
            int length = array.Length;
            for (int i = 0; i < length; i++)
            {
                WriteFloat(stream, array[i]);
            }
        }
        private float[] ReadFloatArray(Stream stream, int length)
        {
            float[] array = new float[length];
            byte[] buffer = new byte[4];
            for (int i = 0; i < length; i++)
            {
                float value = ReadFloat(stream, buffer);
                array[i] = value;
            }
            return array;
        }

        private static void WriteFloat(Stream stream, float value)
        {
            byte[] buffer = BitConverter.GetBytes(value);
            if (!BitConverter.IsLittleEndian)
            {
                byte temp = buffer[0];
                buffer[0] = buffer[3];
                buffer[3] = temp;
                temp = buffer[1];
                buffer[1] = buffer[2];
                buffer[2] = temp;
            }
            stream.Write(buffer);
        }
        private static float ReadFloat(Stream stream, byte[] buffer)
        {
            stream.Read(buffer);
            if (!BitConverter.IsLittleEndian)
            {
                byte temp = buffer[0];
                buffer[0] = buffer[3];
                buffer[3] = temp;
                temp = buffer[1];
                buffer[1] = buffer[2];
                buffer[2] = temp;
            }
            float value = BitConverter.ToSingle(buffer);
            return value;
        }

        private static void WriteInt(Stream stream, int value)
        {
            byte[] buffer = BitConverter.GetBytes(value);
            if (!BitConverter.IsLittleEndian)
            {
                byte temp = buffer[0];
                buffer[0] = buffer[3];
                buffer[3] = temp;
                temp = buffer[1];
                buffer[1] = buffer[2];
                buffer[2] = temp;
            }
            stream.Write(buffer);
        }
        private static int ReadInt(Stream stream, byte[] buffer)
        {
            stream.Read(buffer);
            if (!BitConverter.IsLittleEndian)
            {
                byte temp = buffer[0];
                buffer[0] = buffer[3];
                buffer[3] = temp;
                temp = buffer[1];
                buffer[1] = buffer[2];
                buffer[2] = temp;
            }
            int value = BitConverter.ToInt32(buffer);
            return value;
        }

    }
}