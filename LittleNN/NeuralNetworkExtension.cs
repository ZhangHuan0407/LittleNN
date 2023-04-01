using System;
using System.Collections.Generic;
using System.IO;

namespace LittleNN
{
    /// <summary>
    /// A common, but non-core, method
    /// </summary>
    partial class NeuralNetwork
    {
        [ThreadStatic]
        private static Random m_Random;
        /// <summary>
        /// return a random value in (-1f,1f), without border value
        /// </summary>
        public static float GetRandom()
        {
            if (m_Random is null)
                m_Random = new Random();
            return 1.999998f * (float)m_Random.NextDouble() - 0.999999f;
        }

        /// <summary>
        /// Tarin neural network with data continue numEpochs times
        /// </summary>
        public void Train(IList<StandardData> standardDatas, int numEpochs)
        {
            if (standardDatas.Count < 1)
                throw new ArgumentException($"dataSets count: {standardDatas.Count}");
            for (int i = 0; i < numEpochs; i++)
            {
                for (int j = 0; j < standardDatas.Count; j++)
                {
                    StandardData dataSet = standardDatas[j];
                    ForwardPropagate(dataSet.Inputs);
                    BackPropagate(dataSet.Targets);
                }
            }
        }

        /// <summary>
        /// Tarin neural network with data once
        /// </summary>
        public float Train(StandardData dataSet) => Train(dataSet.Inputs, dataSet.Targets);
        /// <summary>
        /// Tarin neural network with data once
        /// </summary>
		public float Train(float[] input, float[] target)
        {
            this.ForwardPropagate(input);
            this.BackPropagate(target);
            float loss = LossFuntion.MSELoss(this.CopyEval(), target);
            return loss;
        }

        /// <summary>
        /// Current neural network use inputs forward propagate and return calculate result
        /// </summary>
        public float[] Forward(float[] inputs)
        {
            ForwardPropagate(inputs);
            return CopyEval();
        }

        /// <summary>
        /// Serialize <see cref="NeuralNetwork"/> to bin data, and write to target file.
        /// </summary>
        public void SaveTo(string filePath)
        {
            using FileStream fileStream = new FileStream(filePath, FileMode.Create, FileAccess.Write);
            NeuralNetworkModel model = new NeuralNetworkModel();
            model.CopyFrom(this);
            model.Write(fileStream);
        }
        /// <summary>
        /// Read target file and deserialize <see cref="NeuralNetwork"/> from bin data.
        /// </summary>
        public static NeuralNetwork LoadFrom(string filePath)
        {
            using FileStream fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
            NeuralNetworkModel model = new NeuralNetworkModel();
            model.Read(fileStream);
            NeuralNetwork neuralNetwork = new NeuralNetwork();
            model.Override(neuralNetwork);
            return neuralNetwork;
        }
    }
}