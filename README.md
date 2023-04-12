**LittleNN**
- [Introduction](#introduction)
- [Usage](#usage)
  - [Example1 Xor Operator](#example1-xor-operator)
    - [DataSet](#dataset)
    - [Code](#code)
  - [Example2 OCR Number Recognize](#example2-ocr-number-recognize)
    - [DataSet](#dataset-1)
    - [Code](#code-1)
  - [Example3 Serialize](#example3-serialize)
    - [Convert to Bin File](#convert-to-bin-file)
    - [Write \& Read with Stream](#write--read-with-stream)
  - [Train with Multi-Thread](#train-with-multi-thread)
- [Compare with TorchSharp](#compare-with-torchsharp)
  - [Functions](#functions)
  - [Library](#library)
  - [Performance](#performance)
- [Contributions](#contributions)

# Introduction
------------
- I have developed a small cross-platform plugin that implements neural network technology. As TorchSharp has too large storage requirements, I created this repository and the expectation is to provided a neural network minimization implementation.
- I have copy source code from this repository [Neural Networks](https://github.com/trentsartain/Neural-Network), and refactor all of them to improve performance or expand functionality.
  - Specifically, I have replaced double with float, removed the Winform reference, and ensured that the sample can be compiled and run on all platforms.
  - [trentsartain license](/trentsartain/Neural-Network/LICENSE)
- As Microsoft no longer supports Visual Studio 2019, I updated the project TargetFramework to `net5.0`. You can clone the source code and modify it according to your needs.
- All of the code is written in C#, and the project use the MIT LICENSE.
- Multi-thread computing is supported, but [performance](#performance) is lower than TorchSharp.
- If you need faster performance, it is recommended to use other popular deep learning frameworks such as TensorFlow, PyTorch, or Keras.

# Usage

## Example1 Xor Operator
| Input A | Input B | Output |
| :-----: | :-----: | :----- |
|  true   |  true   | false  |
|  true   |  false  | true   |
|  false  |  true   | true   |
|  false  |  false  | false  |

- equal logic : (inputA, inputB) => inputA != inputB;
- Neural Networks accept float value [0, 1] only.

### DataSet
| Input A | Input B | Output |
| :-----: | :-----: | :----- |
|    1    |    1    | 0      |
|    1    |    0    | 1      |
|    0    |    1    | 1      |
|    0    |    0    | 0      |

### Code
- [XorSampleNN.cs](./LittleNN/UnitTest/XorSampleNN.cs)
- Default loss function use MSELoss
- UnitTest.XorSampleNN must Compile with symbol `UNIT_TEST` or `DEBUG`
- Change project compile target to Executable.(default is dll)
![Alt text](Sample/ProjectOptions.png?raw=true "ProjectOptions")
- Test in [Program.cs](./LittleNN/Program.cs)

```
void Main(string[] args)
{
    UnitTest.XorSampleNN.TrainUnitTest();
}
```

## Example2 OCR Number Recognize
- I have 11 symbol texture, 0-9 and '.'
- Take a screenshot, split number pixel's rect with 12x16, and convert to gray value
  $$color(Gray)=average(color(RGBA))$$
- Trains NN recognize number

### DataSet
- input: float[192]
  - the gray value of a number pixel
- target: float[11]
  - example: symbol is '0', float[] { 1, 0, 0,... }

### Code
- [OCRNumberSampleNN.cs](./LittleNN/UnitTest/OCRNumberSampleNN.cs)
- UnitTest.OCRNumberSampleNN must Compile with symbol `UNIT_TEST` or `DEBUG`

```
void Main(string[] args)
{
    // Convert gray value's bytes to string
    UnitTest.OCRNumberSampleNN.WriteNumberGrayValue();
    UnitTest.OCRNumberSampleNN.ConvergeUnitTest();
}
```

- Extract average loss and write in `OCRNumberLoss.csv`, you can create loss statistical data sheet by Excel.
- X-Axis: train times, Y-Axis: MSELoss (the less, the better)
![Alt text](Sample/OCRNumberLoss.png?raw=true "OCRNumberLoss")

## Example3 Serialize
### Convert to Bin File
- [SerizlizeSampleNN.cs](./LittleNN/UnitTest/SerizlizeSampleNN.cs)
- UnitTest.SerizlizeSampleNN must Compile with symbol `UNIT_TEST` or `DEBUG`
- In the fact, due to reduce file size, Neuron.BiasDelta and Synapse.WeightDelta will be loss.
- Serialize and deserizlize is equivalent to set Momentum 0 which made the train of nn slight difference in the first time.
- It will then return to normal.
- Serialization is not designed for piecewise training, but simply storage the parameters of the forward propagate.

```
NeuralNetwork network = new NeuralNetwork(2, new int[] { 6, 6, }, 1);
// train network
network.SaveTo("xxx.bin");
// Load NeuralNetwork from bin file
NeuralNetwork network2 = NeuralNetwork.LoadFrom("xxx.bin");
```
### Write & Read with Stream
- [SerizlizeSampleNN.cs](./LittleNN/UnitTest/SerizlizeSampleNN.cs)
- UnitTest.SerizlizeSampleNN must Compile with symbol `UNIT_TEST` or `DEBUG`
- NeuralNetwork.SaveTo, NeuralNetwork.LoadFrom implement by NeuralNetworkModel.

```
NeuralNetworkModel model = new NeuralNetworkModel();
model.CopyFrom(network);
using (MemoryStream memoryStream = new MemoryStream())
{
    // serialize
    model.Write(memoryStream);
    // deserialize
    model = new NeuralNetworkModel();
    model.Read(memoryStream);
}
```

## Train with Multi-Thread
```
neuralNetwork.SetMultiThread(true);
neuralNetwork.Train(xxx, xxx);
neuralNetwork.Forward(xxx);
neuralNetwork.SetMultiThread(false);
```

- If you forget to invoke NeuralNetwork.SetMultiThread(false), inner thread will be revert to CalculateThread.IdleThread when GC invoke NeuralNetwork.Finalize
```
~NeuralNetwork()
{
    SetMultiThread(false);
}
```

- Only one thread will be cache at CalculateThread.IdleThread and waiting for NeuralNetwork.SetMultiThread(true) resume it, superfluous thread will be abort.
- If you train hundreds of NeuralNetwork in memory and both want to use multi thread (Suppose 300,300,300 NeuralNetwork, 90 Mb and 1 thread for each NeuralNetwork instance),
- You'd better manual set thread instance.
- The instance of CalculateThread can't be reference by different instance of NeuralNetwork in the same time.
```
CalculateThread calculateThread = CalculateThread.IdleThread; // get recycle thread or new thread
for (int i = 0; i < 100; i++)
{
    neuralNetworkList[i].CalculateThread = calculateThread;   // replace SetMultiThread(true)
    neuralNetworkList[i].Train(xxx, xxx);
    neuralNetworkList[i].CalculateThread = null;              // replace SetMultiThread(false)
}
calculateThread.Abort();                                      // mark thread abort, it will run to exit
```

# Compare with TorchSharp
## Functions
- LittleNN support Activation.Sigmoid only, there are many fewer functions to choose from than TourchSharp.
- If you want a powerful library for learning neural network, I recommend TourchSharp.

## Library
|                      | [TourchSharp-cpu](https://github.com/dotnet/TorchSharp) | LittleNN         |
| -------------------- | ------------------------------------------------------- | ---------------- |
| Architecture         | X86-64✓ Arm64x                                          | .net runtime 5.0 |
| Windows Library Size | torch_cpu.dll 240MB, are you kidding me?                | less than 30KB   |
| OSX Library Size     | torch_cpu.dylib 390MB                                   | less than 30KB   |
| Linix Library Size   | torch_cpu.so 520MB                                      | less than 30KB   |

## Performance
- Example2 OCR Number Recognize
- Caculate duration, 200 times Network.Forward() total
- MacOS, 2.6 GHz 6-Core Intel Core i7
- LittleNN NeuralNetworkModel.QuickForward replace 'Neural and Synapse instance' with 'float[]'. Due to element of float[] are continuous on the Memory, NeuralNetworkModel.QuickForward is faster than NeuralNetwork.Forward.
- If estimated operational volume less than CalculateThread.AmountOfComputation, NeuralNetwork calculate in single thread.
  - Estimated operational volume = neural count of A layer *  neural count of B layer * 10
  - CalculateThread.AmountOfComputation = 6400 * 10
- In unit test, NeuralNetworkModel.QuickForward with multithreading have performance improvement less than 5%, so I dropped support for NeuralNetworkModel Multithreading.
- In test(192,100,100,11), thread synchronization wastes more time than the computing boost from multithreading.

| InputSize,HideSize,OutputSize | TourchSharp-cpu | NeuralNetwork.SingleThread | NeuralNetwork.MultiThread | NeuralNetworkModel.QuickForward |
| ----------------------------- | --------------- | -------------------------- | ------------------------- | ------------------------------- |
| 192,20,20,11                  | ≈13ms           | ≈2ms                       | \                         | ≈4ms                            |
| 192,30,30,11                  | ≈14ms           | ≈3ms                       | \                         | ≈4ms                            |
| 192,100,100,11                | ≈16ms           | ≈12ms                      | ≈19ms                     | ≈13ms                           |
| 192,300,300,300,11            | ≈21ms           | ≈138ms                     | ≈92ms                     | ≈73ms                           |

# Contributions
- LittleNN is enough for beginner.
- Contributions and feedback are welcome. If you have any suggestions or bugs you want to report, please open an issue or pull request.