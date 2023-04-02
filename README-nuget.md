**LittleNN**
- [Introduction](#introduction)
- [Usage](#usage)
  - [Example1 Xor Operator](#example1-xor-operator)
    - [Code](#code)
  - [Example2 OCR Number Recognize](#example2-ocr-number-recognize)
  - [Example3 Serialize](#example3-serialize)

# Introduction
------------
- The expectation is to provided a neural network minimization implementation.
- TargetFramework is `net5.0`. You can modify it according to your needs.
- All of the code is written in C#, and the project use the MIT LICENSE.
- Get more info from repository [README](https://github.com/ZhangHuan0407/LittleNN).
- If you need faster performance, it is recommended to use other popular deep learning frameworks such as TensorFlow, PyTorch, or Keras.

# Usage

## Example1 Xor Operator
| Input A | Input B | Output |
| :-----: | :-----: | :----- |
|    1    |    1    | 0      |
|    1    |    0    | 1      |
|    0    |    1    | 1      |
|    0    |    0    | 0      |

### Code
```
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
    StandardData dataSet = dataSets[random.Next() % dataSets.Length];
    float error = network.Train(dataSet);
    if (i % (20 * 1000) == 0)
        // Expect: The error will decrease gradually
        Console.WriteLine($"{i}   : {error}");
}
```

## Example2 OCR Number Recognize
- Get more info from repository [README](https://github.com/ZhangHuan0407/LittleNN).

## Example3 Serialize
- Get more info from repository [README](https://github.com/ZhangHuan0407/LittleNN).