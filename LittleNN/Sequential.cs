using System;
using System.Collections.Generic;

namespace LittleNN
{
    /// <summary>
    /// Neural network contains layers and synapse whitch between two neural.
    /// <para>This class wanna to declare a neural network decided and fully.</para>
    /// <para>See more example in XorSampleNN2.cs</para>
    /// </summary>
    public class Sequential
    {
        /// <summary>
        /// The type of <see cref="Sequential"/>
        /// </summary>
        public enum SequentialParameter
        {
            /// <summary>
            /// use <see cref="NeuralCount"/> field
            /// </summary>
            LayerNeuralCount,
            /// <summary>
            /// use <see cref="ActType"/>, <see cref="ActParameter"/> fields
            /// </summary>
            ActivationFunction,
        }
        /// <summary>
        /// just annotation, no practical use
        /// </summary>
        public string Annotation;
        /// <summary>
        /// The type of <see cref="Sequential"/>
        /// </summary>
        public SequentialParameter ParameterType;
        /// <summary>
        /// neural count in this layer
        /// </summary>
        public int NeuralCount;
        /// <summary>
        /// One of <see cref="ActivationsFunctionType"/>, or your custom type: (ActivationsFunctionType)1001
        /// </summary>
        public ActivationsFunctionType ActType;
        /// <summary>
        /// The parameter of activation function, default(null) will be convert to 0f
        /// </summary>
        public float? ActParameter;

        /// <summary>
        /// Create an empty instance
        /// </summary>
        protected Sequential()
        {
        }

        /// <summary>
        /// Declare a neural layers with specified number
        /// </summary>
        /// <param name="annotation">just annotation, no practical use</param>
        /// <param name="neuralCount">neural count in this layer</param>
        public static Sequential Neural(string annotation, int neuralCount)
        {
            Sequential sequential = new Sequential()
            {
                Annotation = annotation,
                ParameterType = SequentialParameter.LayerNeuralCount,
                NeuralCount = neuralCount,
            };
            return sequential;
        }
        /// <summary>
        /// Declare a specified activation function between two layers
        /// </summary>
        /// <param name="annotation">just annotation, no practical use</param>
        /// <param name="actType">One of <see cref="ActivationsFunctionType"/>, or your custom type: (ActivationsFunctionType)1001</param>
        /// <param name="actParameter">The parameter of activation function, default(null) will be convert to 0f</param>
        public static Sequential Activation(string annotation, ActivationsFunctionType actType, float? actParameter = null)
        {
            Sequential sequential = new Sequential()
            {
                Annotation = annotation,
                ParameterType = SequentialParameter.ActivationFunction,
                ActType = actType,
                ActParameter = actParameter,
            };
            return sequential;
        }

        /// <summary>
        /// Create an empty list
        /// <para>Please add <see cref="Sequential"/> instance which create by <see cref="Sequential.Neural"/> and <see cref="Sequential.Activation"/></para>
        /// <para>Each neural network must have more than three layers</para>
        /// <para>Default activation is <see cref="ActivationsFunctionType.Sigmoid"/></para>
        /// </summary>
        public static List<Sequential> CreateNew() => new List<Sequential>();
    }
    /// <summary>
    /// Neural network contains layers and synapse whitch between two neural.
    /// <para>This class wanna to declare a neural network decided and fully.</para>
    /// <para>See more example in XorSampleNN2.cs</para>
    /// </summary>
    public static class SequentialExtension
    {
        /// <summary>
        /// Calculate layers count in this list
        /// </summary>
        public static int LayerCount(this List<Sequential> sequentials)
        {
            int count = 0;
            for (int i = 0; i < sequentials.Count; i++)
            {
                if (sequentials[i].ParameterType == Sequential.SequentialParameter.LayerNeuralCount)
                    count++;
            }
            return count;
        }
    }
}