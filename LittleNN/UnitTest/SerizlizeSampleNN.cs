#if DEBUG || UNIT_TEST
using System;
using System.Collections.Generic;
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
						throw new Exception("not equal! EvalValueList.");
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

            for (int i = 0; i < network.HiddenLayers.Length; i++)
            {
                if (network.HiddenLayers[i].ActParameter != network2.HiddenLayers[i].ActParameter ||
					network.HiddenLayers[i].ActType != network2.HiddenLayers[i].ActType)
					throw new Exception("not equal! ActParameter or ActType.");
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
						throw new Exception("not equal! EvalValueList3.");
				}
            }

			Console.WriteLine("equal");
		}

#pragma warning disable CS1591
		public static void IOUnitTest_CustomParameter()
		{
			Console.WriteLine("SerizlizeSampleNN.IOUnitTest_CustomParameter");
			Random random = new Random();

			(ActivationsFunctionType, float?)[] ActParameters = new (ActivationsFunctionType, float?)[]
			{
				(ActivationsFunctionType.LeakyReLU, 0.01f),
				(ActivationsFunctionType.LeakyReLU, 0.02f),
				// ReLU will make loss NaN, so i ignore this ActivationsFunctionType
				// I guess no one uses it?
				//(ActivationsFunctionType.ReLU, null),
				(ActivationsFunctionType.Sigmoid, null),
				(ActivationsFunctionType.Softsign, null),
			};

            for (int times = 0; times < 1000; times++)
            {
                if (times % 100 == 99)
					Console.WriteLine("random create nn and serialzie " + times);
				List<Sequential> sequentials = Sequential.CreateNew();
				sequentials.Add(Sequential.Neural("input layer", random.Next() % 5 + 10));
				int calculationLayerCount = random.Next() % 3 + 2;
				for (int i = 0; i < calculationLayerCount; i++)
				{
					(ActivationsFunctionType, float?) actParameter = ActParameters[random.Next() % ActParameters.Length];
					sequentials.Add(Sequential.Activation("activation " + i, actParameter.Item1, actParameter.Item2));
					if (i == calculationLayerCount - 1)
						sequentials.Add(Sequential.Neural("output layer", 9));
					else
						sequentials.Add(Sequential.Neural("hidden layer " + i, random.Next() % 20 + 2));
				}
				NeuralNetwork neuralNetworkTemplate = new NeuralNetwork(sequentials, (float)random.NextDouble() % 0.05f + 0.01f, (float)random.NextDouble() % 0.6f);
				NeuralNetworkModel model = new NeuralNetworkModel();
				model.CopyFrom(neuralNetworkTemplate);
				NeuralNetwork neuralNetworkMirror = new NeuralNetwork();
				model.Override(neuralNetworkMirror);

                for (int trainTimes = 0; trainTimes < 5; trainTimes++)
                {
					// input is binary number
					float[] input = NeuralNetwork.RandomN(neuralNetworkTemplate.InputLayer.NeuronsCount);
					int[] inputInt = new int[10];
					for (int i = 0; i < 10; i++)
                    {
						inputInt[i] = random.Next() % 2;
						input[i] = inputInt[i];
                    }
					// target is xor result
					float[] target = new float[9];
                    for (int i = 0; i < 9; i++)
						target[i] = inputInt[i] ^ inputInt[i + 1];
					float lossTemplate = neuralNetworkTemplate.Train(input, target);
					float lossMirror = neuralNetworkMirror.Train(input, target);
					if (lossTemplate != lossMirror || lossTemplate == 0f)
						throw new Exception("NeuralNetwork is not sync");
                }
			}
			Console.WriteLine("all equal");
		}
#pragma warning restore CS1591
	}
}
#endif