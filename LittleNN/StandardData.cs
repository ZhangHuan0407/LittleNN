using System;

namespace LittleNN
{
    /// <summary>
    /// Data format of <see cref="NeuralNetwork.Train(StandardData)"/>,
    /// but use <see cref="StandardData"/> is not necessary.
	/// <para>This class just a wrapper of input and target's data</para>
    /// </summary>
    public class StandardData
    {
        /// <summary>
        /// input data array, each element in range(0, 1)
        /// </summary>
        public readonly float[] Inputs;
        /// <summary>
        /// target data array, each element in range(0, 1)
        /// </summary>
        public readonly float[] Targets;

        /// <summary>
        /// Pack input and target's data
        /// </summary>
        public StandardData(float[] inputs, float[] targets)
        {
            Inputs = inputs;
            Targets = targets;
        }
    }
}