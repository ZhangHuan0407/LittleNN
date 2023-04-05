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
        private static readonly object LockObject;
        /// <summary>
        /// Avoid seed synchronization
        /// </summary>
        private static int RandomOffset;
        [ThreadStatic]
        private static Random m_Random;
        /// <summary>
        /// return a random value in (-1f,1f), without border value
        /// </summary>
        public static float GetRandom()
        {
            if (m_Random is null)
            {
                lock (LockObject)
                {
                    m_Random = new Random(RandomOffset);
                    RandomOffset = m_Random.Next() + m_Random.Next();
                }
            }
            return 1.999998f * (float)m_Random.NextDouble() - 0.999999f;
        }
        /// <summary>
        /// Create a random value array with specified length
        /// </summary>
        /// <param name="length">the length of random value array</param>
        public static float[] RandomN(int length)
        {
            if (m_Random is null)
            {
                lock (LockObject)
                {
                    m_Random = new Random(RandomOffset);
                    RandomOffset = m_Random.Next() + m_Random.Next();
                }
            }
            float[] result = new float[length];
            for (int i = 0; i < length; i++)
                result[i] = (float)m_Random.NextDouble();
            return result;
        }

        static NeuralNetwork()
        {
            LockObject = new object();
            m_Random = new Random();
            RandomOffset = m_Random.Next();
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
                    OptimizerBackward(dataSet.Targets);
                    OptimizerStep();
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
            this.OptimizerBackward(target);
            this.OptimizerStep();
            float loss = LossFuntion.MSELoss(this.CopyEvaluation(), target);
            return loss;
        }

        /// <summary>
        /// Current neural network use inputs forward propagate and return calculate result
        /// </summary>
        public float[] Forward(float[] inputs)
        {
            ForwardPropagate(inputs);
            return CopyEvaluation();
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