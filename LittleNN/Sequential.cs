using System;
using System.Collections.Generic;

namespace LittleNN
{
    public class Sequential
    {
        public enum SequentialParameter
        {
            LayerNeuralCount,
            ActivationFunction,
        }
        public string Annotation;
        public SequentialParameter ParameterType;
        public int NeuralCount;
        public ActivationsFunctionType ActType;
        public float? ActParameter;

        protected Sequential()
        {
        }

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

        public static List<Sequential> CreateNew() => new List<Sequential>();
    }
    public static class SequentialExtension
    {
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