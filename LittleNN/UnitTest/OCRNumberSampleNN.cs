#if DEBUG || UNIT_TEST
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using LittleNN;
// using SkiaSharp;

namespace UnitTest
{
    internal static class OCRNumberSampleNN
    {
        public const int Width = 12;
        public const int Height = 16;
        public const int StdLength = 12 * 16;

        private static Dictionary<int, List<float[]>> LoadDatasSet()
        {
            Dictionary<int, List<float[]>> result = new Dictionary<int, List<float[]>>();
            if (!Directory.Exists("./UnitTest/NumberBytes"))
            {
                throw new Exception("You don't have unit test data...\nCopy LittleNN/UnitTest/NumberBytes to bin/Debug?/UnitTest/NumberBytes");
            }
            foreach (string dictionaryPath in Directory.GetDirectories("./UnitTest/NumberBytes"))
            {
                string name = new DirectoryInfo(dictionaryPath).Name;
                int sampleValue;
                if (name[0] == '_')
                    sampleValue = 10;
                else
                    sampleValue = name[0] - '0';
                List<float[]> samplesList = new List<float[]>();
                result.Add(sampleValue, samplesList);
                foreach (string filePath in Directory.GetFiles(dictionaryPath))
                {
                    // fuck MacOS .DS_Store
                    if (filePath.EndsWith(".DS_Store"))
                    {
                        File.Delete(filePath);
                        continue;
                    }
                    byte[] bytes = File.ReadAllBytes(filePath);
                    if (bytes.Length != StdLength)
                        throw new Exception();
                    float[] sample = new float[bytes.Length];
                    for (int i = 0; i < bytes.Length; i++)
                        sample[i] = bytes[i] / 255f;
                    samplesList.Add(sample);
                }
            }
            return result;
        }

        /// <summary>
        /// See more info in README.md
        /// </summary>
        public static void ConvergeUnitTest()
        {
            Console.WriteLine("OCRNumberSampleNN.ConvergeUnitTest");
            // key: number sequence id, from 0 to 11
            // value: number bytes (draw alphabet texture, split font rect, convert to grey value and encoding to byte[])
            // each bytes.Length is equal StdLength 
            Dictionary<int, List<float[]>> DatasSet = LoadDatasSet();
            NeuralNetwork network = new NeuralNetwork(StdLength, new int[] { 100, 100, }, 11);

            const int NumEpochs = 3000;

            StringBuilder lossStringBuilder = new StringBuilder();
            for (int i = 0; i < NumEpochs; i += 10)
                lossStringBuilder.Append(i).Append(',');
            lossStringBuilder.Length -= 1;
            lossStringBuilder.Append('\n');

            Random random = new Random();
            Stopwatch trainStopWatch = Stopwatch.StartNew();
            for (int i = 0; i < NumEpochs; i++)
            {
                float totalLoss = 0f;
                foreach (KeyValuePair<int, List<float[]>> pair in DatasSet)
                {
                    float[] input = pair.Value[random.Next() % pair.Value.Count];
                    float[] target = new float[11];
                    target[pair.Key] = 1f;
                    float loss = network.Train(input, target);
                    totalLoss += loss;
                }
                if (i % 10 == 0)
                {
                    lossStringBuilder.Append($"{totalLoss / DatasSet.Count}").Append(',');
                }
                if (i % 200 == 199)
                {
                    Console.WriteLine($"{i}  :{totalLoss / DatasSet.Count} {trainStopWatch.ElapsedMilliseconds}");
                    trainStopWatch.Restart();
                }
            }
            lossStringBuilder.Length -= 1;
            lossStringBuilder.Append('\n');
            File.WriteAllText("OCRNumberLoss.csv", lossStringBuilder.ToString());
            Console.WriteLine(Path.GetFullPath("OCRNumberLoss.csv"));

            Stopwatch stopwatch = Stopwatch.StartNew();
            for (int i = 0; i < 200; i++)
            {
                int sampleIndex = random.Next() % 11;
                List<float[]> list = DatasSet[sampleIndex];
                network.Forward(list[random.Next() % list.Count]);
            }
            Console.WriteLine($"200 forward in {stopwatch.ElapsedMilliseconds} ms");
        }

        public static void WriteNumberGrayValue()
        {
            Console.WriteLine("OCRNumberSampleNN.WriteNumberGrayValue");
            Dictionary<int, List<float[]>> DatasSet = LoadDatasSet();
            foreach (KeyValuePair<int, List<float[]>> pair in DatasSet)
            {
                Console.WriteLine(pair.Key);
                float[] template = pair.Value[0];
                for (int i = 0; i < template.Length; i++)
                {
                    int value = (int)(template[i] * 255f);
                    Console.Write(value.ToString().PadLeft(4));
                    if (i % Width == Width - 1)
                        Console.WriteLine();
                }
            }
        }

        /*
        /// <summary>
        /// split number rect from Bitmap
        /// data conversion, is not neccesary with NN
        /// </summary>
        /// <param name="filePath">Bitmap file path</param>
        static void CreateGraySample(string filePath)
        {
            SKBitmap subBitmap;
            SKCanvas canvas;
            using (SKBitmap bitmap = SKBitmap.Decode(filePath))
            {
                // hard code coordinate is the simplest
                Rectangle localRect = RectAnchor.InMapTime.CalculateLocalRectangle(new Rectangle(0, 0, bitmap.Width, bitmap.Height));
                subBitmap = new SKBitmap(localRect.Width, localRect.Height);
                canvas = new SKCanvas(subBitmap);
                canvas.DrawBitmap(bitmap,
                                  source: new SKRect(localRect.Left, localRect.Top, localRect.Right, localRect.Bottom),
                                  dest: new SKRect(0, 0, localRect.Width, localRect.Height));
            }
            GrayMap grayMap = GrayMap.CreateGrayMap(subBitmap.Info, subBitmap.GetPixelSpan());

            // debug rect
            //{
            //    using SKPaint paint = new SKPaint();
            //    paint.Color = new SKColor(200, 0, 0);
            //    for (int i = 0; i < grayMap.SubAreaRect.Count; i++)
            //    {
            //        var rect = grayMap.SubAreaRect[i];
            //        canvas.DrawLine(new SKPoint(rect.Left, rect.Top), new SKPoint(rect.Right, rect.Bottom), paint);
            //    }
            //    using (Stream stream = new FileStream("aa.png", FileMode.Create, FileAccess.Write))
            //    {
            //        subBitmap.Encode(stream, SKEncodedImageFormat.Png, 0);
            //    }
            //    canvas.Dispose();
            //}
            subBitmap.Dispose();

            Directory.CreateDirectory("../../../Text");
            Directory.CreateDirectory("../../../Bytes");
            Random random = new Random();
            byte[] buffer = new byte[GrayMap.StdLength];
            for (int i = 0; i < grayMap.SubAreaRect.Count; i++)
            {
                StringBuilder stringBuilder = new StringBuilder();
                grayMap.CopyRawValue(grayMap.SubAreaRect[i], ref buffer);
                for (int point = 0; point < buffer.Length; point++)
                {
                    stringBuilder.Append(buffer[point].ToString().PadLeft(5));
                    if (point % GrayMap.StdSubAreaSize.Width == GrayMap.StdSubAreaSize.Width - 1)
                    {
                        stringBuilder.AppendLine();
                    }
                }
                stringBuilder.AppendLine();
                int id = Math.Abs(random.Next() % 100000);
                File.WriteAllText($"../../../Text/{id}.txt", stringBuilder.ToString());
                File.WriteAllBytes($"../../../Bytes/{id}.bin", buffer);
            }
        }
        */

    }

    /*
    // data conversion, is not neccesary with NN
    public struct GrayMap
    {
        // 由于图片上的数字大小是相对固定的(经验值)
        public static readonly SKSizeI StdSubAreaSize = new SKSizeI(OCRNumberSampleNN.Width, OCRNumberSampleNN.Height);

        public byte[,] Value;

        public List<SKRectI> SubAreaRect;

        public static GrayMap CreateGrayMap(SKImageInfo info, ReadOnlySpan<byte> bytes)
        {
            int width = info.Width;
            int height = info.Height;
            int pixelBytesCount = info.BytesPerPixel;
            if (pixelBytesCount != 4 ||
                info.ColorType != SKColorType.Rgba8888 && info.ColorType != SKColorType.Bgra8888)
                throw new Exception();
            GrayMap result = default;
            result.Value = new byte[height, width];
            long totalSum = 0L;
            for (int y = 0; y < height; y++)
            {
                int point = y * width * 4;
                for (int x = 0; x < width; x++)
                {
                    byte q = bytes[point++];
                    byte w = bytes[point++];
                    byte e = bytes[point++];
                    point++;
                    byte average = (byte)((q + w + e) / 3);
                    result.Value[y, x] = average;
                    totalSum += average;
                }
            }
            byte lowValue = (byte)(0.4f * totalSum / width / height);
            byte hightValue = (byte)(0.5f * totalSum / width / height + 127.5f);

            bool[,] havePicked = new bool[height, width];
            result.SubAreaRect = new List<SKRectI>();
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    if (!havePicked[y, x] && result.Value[y, x] > hightValue)
                    {
                        SKRectI rect = new SKRectI(x, y, x, y);
                        Pick(x, y, ref rect);
                        if (rect.Width > 0 && rect.Width < StdSubAreaSize.Width &&
                            rect.Height > 0 && rect.Height < StdSubAreaSize.Height)
                        {
                            rect.Right++;
                            rect.Bottom++;
                            result.SubAreaRect.Add(rect);
                        }
                    }
            void Pick(int pointX, int pointY, ref SKRectI subAreaRect)
            {
                if (pointX < 0 || pointY < 0 || pointX >= width || pointY >= height)
                    return;
                else if (havePicked[pointY, pointX] || result.Value[pointY, pointX] < lowValue)
                    return;
                if (pointX < subAreaRect.Left)
                    subAreaRect.Left = pointX;
                if (pointX > subAreaRect.Right)
                    subAreaRect.Right = pointX;
                if (pointY < subAreaRect.Top)
                    subAreaRect.Top = pointY;
                if (pointY > subAreaRect.Bottom)
                    subAreaRect.Bottom = pointY;
                havePicked[pointY, pointX] = true;
                Pick(pointX - 1, pointY, ref subAreaRect);
                Pick(pointX + 1, pointY, ref subAreaRect);
                Pick(pointX, pointY - 1, ref subAreaRect);
                Pick(pointX, pointY + 1, ref subAreaRect);
                if (result.Value[pointY, pointX] > hightValue)
                {
                    Pick(pointX - 1, pointY - 1, ref subAreaRect);
                    Pick(pointX - 1, pointY + 1, ref subAreaRect);
                    Pick(pointX + 1, pointY - 1, ref subAreaRect);
                    Pick(pointX + 1, pointY - 1, ref subAreaRect);
                }
            }

            //File.WriteAllText("log.txt", stringBuilder.ToString());

            return result;
        }

        public void CopyRawValue(SKRectI subAreaRect, ref byte[] buffer)
        {
            Array.Clear(buffer, 0, OCRNumberSampleNN.StdLength);
            int offsetX = (StdSubAreaSize.Width - subAreaRect.Width) / 2;
            int offsetY = (StdSubAreaSize.Height - subAreaRect.Height) / 2;
            int left = subAreaRect.Left;
            int top = subAreaRect.Top;
            for (int y = 0; y < subAreaRect.Height; y++)
                for (int x = 0; x < subAreaRect.Width; x++)
                {
                    int point = (offsetY + y) * StdSubAreaSize.Width + (offsetX + x);
                    buffer[point] = Value[top + y, left + x];
                }
        }
    }
    */
}
#endif