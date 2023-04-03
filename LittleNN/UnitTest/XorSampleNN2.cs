#if DEBUG || UNIT_TEST
using System;
using System.Collections.Generic;
using LittleNN;

namespace UnitTest
{
    public static class XorSampleNN2
	{
		public static void ReLUUnitTest()
		{
			Console.WriteLine("XorSampleNN2.ReLUUnitTest");
			StandardData[] dataSets = new StandardData[]
			{
				new StandardData(new float[] { 1f, 1f }, new float[] { 0f }),
				new StandardData(new float[] { 0f, 1f }, new float[] { 1f }),
				new StandardData(new float[] { 1f, 0f }, new float[] { 1f }),
				new StandardData(new float[] { 0f, 0f }, new float[] { 0f }),
			};

            List<Sequential> sequential = Sequential.CreateNew();
			sequential.Add(Sequential.Neural("input layer", 2));
			sequential.Add(Sequential.Activation("first link", ActivationsFunctionType.ReLU));
			sequential.Add(Sequential.Neural("hidden layer 1", 6));
			sequential.Add(Sequential.Activation("second link", ActivationsFunctionType.Sigmoid));
			sequential.Add(Sequential.Neural("hidden layer 2", 6));
			sequential.Add(Sequential.Activation("third link", ActivationsFunctionType.Sigmoid));
			sequential.Add(Sequential.Neural("output layer", 1));

			NeuralNetwork network = new NeuralNetwork(2, new int[] { 6, 6, }, 1);

			Random random = new Random();
			for (int i = 0; i < 500 * 1000; i++)
			{
				StandardData dataSet = dataSets[random.Next() % dataSets.Length];
				float error = network.Train(dataSet);
				if (i % (10 * 1000) == 0)
					Console.WriteLine($"{i}   : {error}");
			}
		}
	}
}
#endif