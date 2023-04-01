#if DEBUG || UNIT_TEST
using System;
using System.IO;
using LittleNN;

namespace UnitTest
{
    internal static class SerizlizeSampleNN
	{
		/// <summary>
		/// See more info in README.md
		/// </summary>
		public static void IOUnitTest()
		{
			Console.WriteLine("SerizlizeSampleNN.IOUnitTest");
			StandardData[] dataSets = new StandardData[]
			{
				new StandardData(new float[] { 1f, 1f }, new float[] { 0f }),
				new StandardData(new float[] { 0f, 1f }, new float[] { 1f }),
				new StandardData(new float[] { 1f, 0f }, new float[] { 1f }),
				new StandardData(new float[] { 0f, 0f }, new float[] { 0f }),
			};
			NeuralNetwork network;
			float[] evalValueList;
			// train a NeuralNetwork the same as XorSampleNN
			{
				network = new NeuralNetwork(2, new int[] { 6, 6, }, 1);
				Random random = new Random();
				for (int i = 0; i < 200 * 1000; i++)
				{
					StandardData dataSet = dataSets[random.Next() % dataSets.Length];
					float error = network.Train(dataSet);
					if (i % (20 * 1000) == 0)
						Console.WriteLine($"{i}   : {error}");
				}

				evalValueList = new float[]
				{
					network.Forward(dataSets[0].Inputs)[0],
					network.Forward(dataSets[1].Inputs)[0],
					network.Forward(dataSets[2].Inputs)[0],
					network.Forward(dataSets[3].Inputs)[0],
				};
			}

			const string filePath = "IOUnitTest.bin";
			network.SaveTo(filePath);
            if (!File.Exists(filePath))
				throw new Exception("File is not exists");
			// get deserizlize NeuralNetwork instance
			NeuralNetwork network2 = NeuralNetwork.LoadFrom(filePath);

			// check serialize and deserizlize instance are value equal
			// deserizlize don't need this step
			{
				float[] evalValueList2 = new float[]
				{
					network2.Forward(dataSets[0].Inputs)[0],
					network2.Forward(dataSets[1].Inputs)[0],
					network2.Forward(dataSets[2].Inputs)[0],
					network2.Forward(dataSets[3].Inputs)[0],
				};
				for (int i = 0; i < evalValueList2.Length; i++)
				{
					if (evalValueList[i] != evalValueList2[i])
						throw new Exception("not equal");
				}
			}
			Random random22 = new Random();
			for (int i = 0; i < 40 * 1000; i++)
			{
				StandardData dataSet = dataSets[random22.Next() % dataSets.Length];
				float error = network2.Train(dataSet);
				if (i % (5 * 1000) == 0)
					Console.WriteLine($"{i}   : {error}");
			}

			// another way to QuickForward
			// NeuralNetworkModel.QuickForward is faster than NeuralNetwork.ForwardPropagate,
			// but NeuralNetworkModel don't have BackPropagate.
			{
				NeuralNetworkModel model = new NeuralNetworkModel();
				model.CopyFrom(network);
				using (MemoryStream memoryStream = new MemoryStream())
				{
					// serialize
					model.Write(memoryStream);
					memoryStream.Seek(0, SeekOrigin.Begin);
					// deserialize
					model = new NeuralNetworkModel();
					model.Read(memoryStream);
				}
				float[] evalValueList3 = new float[]
				{
					model.QuickForward(dataSets[0].Inputs)[0],
					model.QuickForward(dataSets[1].Inputs)[0],
					model.QuickForward(dataSets[2].Inputs)[0],
					model.QuickForward(dataSets[3].Inputs)[0],
				};
				for (int i = 0; i < evalValueList3.Length; i++)
				{
					if (evalValueList[i] != evalValueList3[i])
						throw new Exception("not equal");
				}
            }

			Console.WriteLine("equal");
		}
	}
}
#endif