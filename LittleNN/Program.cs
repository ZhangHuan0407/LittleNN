﻿#if DEBUG || UNIT_TEST
using System;

namespace LittleNN
{
    internal static class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            // invoke one of SampleNN
            //UnitTest.XorSampleNN.TrainUnitTest();
            //UnitTest.OCRNumberSampleNN.ConvergeUnitTest();
            //UnitTest.OCRNumberSampleNN.WriteNumberGrayValue();
            //UnitTest.SerizlizeSampleNN.IOUnitTest();
            //UnitTest.SerizlizeSampleNN.IOUnitTest_CustomParameter();
            //UnitTest.ActivationFunctionUnitTest.MathUnitTest();
            //UnitTest.XorSampleNN2.ReLUUnitTest();
            //UnitTest.MultiThreadUnitTest.ThreadFunctionUnitTest();

        }
    }
}
#endif