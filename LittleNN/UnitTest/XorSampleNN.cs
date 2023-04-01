#if DEBUG || UNIT_TEST
using System;
using System.Collections.Generic;
using System.IO;
using LittleNN;

namespace UnitTest
{
	internal static class XorSampleNN
    {
        /// <summary>
        /// See more info in README.md
        /// </summary>
        public static void TrainUnitTest()
		{
			Console.WriteLine("XorSampleNN.TrainUnitTest");
			// This network with single hidden layer probably not converge in limited times
			// Network network = new Network(2, new int[] { 2, }, 1);

			NeuralNetwork network = new NeuralNetwork(2, new int[] { 6, 6, }, 1);

			StandardData[] dataSets = new StandardData[]
			{
				new StandardData(new float[] { 1f, 1f }, new float[] { 0f }),
				new StandardData(new float[] { 0f, 1f }, new float[] { 1f }),
				new StandardData(new float[] { 1f, 0f }, new float[] { 1f }),
				new StandardData(new float[] { 0f, 0f }, new float[] { 0f }),
			};

			Random random = new Random();
            for (int i = 0; i < 1000 * 1000; i++)
            {
				// This sample code random pick one of data
				// You should better foreach all of data
                StandardData dataSet = dataSets[random.Next() % dataSets.Length];
				float error = network.Train(dataSet);
                if (i % (20 * 1000) == 0)
					// Expect: The error will decrease gradually
					Console.WriteLine($"{i}   : {error}");
			}
        }
	}
}
#endif