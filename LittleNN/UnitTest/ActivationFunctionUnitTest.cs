#if DEBUG || UNIT_TEST
using System;
using LittleNN;

namespace UnitTest
{
    public static class ActivationFunctionUnitTest
    {
        /// <summary>
        /// https://keisan.casio.com/exec/system/15236005297459
        /// </summary>
        public static void MathUnitTest()
        {
            Console.WriteLine("MathUnitTest");
            AssertNearly(ActivationsFunctions.Output(ActivationsFunctionType.LeakyReLU, 0f, 0f), 0f);
            AssertNearly(ActivationsFunctions.Output(ActivationsFunctionType.LeakyReLU, 0.5f, 0f), 0.5f);
            AssertNearly(ActivationsFunctions.Output(ActivationsFunctionType.LeakyReLU, 1f, 0f), 1f);
            AssertNearly(ActivationsFunctions.Output(ActivationsFunctionType.LeakyReLU, -0.5f, 0f), 0f);
            AssertNearly(ActivationsFunctions.Output(ActivationsFunctionType.LeakyReLU, -1f, 0f), 0f);
            AssertNearly(ActivationsFunctions.Output(ActivationsFunctionType.LeakyReLU, -0.5f, 0.02f), -0.01f);
            AssertNearly(ActivationsFunctions.Output(ActivationsFunctionType.LeakyReLU, -1f, 0.02f), -0.02f);

            AssertNearly(ActivationsFunctions.Output(ActivationsFunctionType.ReLU, 0f, 0f), 0f);
            AssertNearly(ActivationsFunctions.Output(ActivationsFunctionType.ReLU, 0.5f, 0f), 0.5f);
            AssertNearly(ActivationsFunctions.Output(ActivationsFunctionType.ReLU, 1f, 0f), 1f);
            AssertNearly(ActivationsFunctions.Output(ActivationsFunctionType.ReLU, -0.5f, 0f), 0f);
            AssertNearly(ActivationsFunctions.Output(ActivationsFunctionType.ReLU, -1f, 0f), 0f);

            AssertNearly(ActivationsFunctions.Output(ActivationsFunctionType.Sigmoid, 0f, 0f), 0.5f);
            AssertNearly(ActivationsFunctions.Output(ActivationsFunctionType.Sigmoid, 0.4f, 0f), 1f / (1f + MathF.Exp(-0.4f)));
            AssertNearly(ActivationsFunctions.Output(ActivationsFunctionType.Sigmoid, 0.8f, 0f), 1f / (1f + MathF.Exp(-0.8f)));
            AssertNearly(ActivationsFunctions.Output(ActivationsFunctionType.Sigmoid, -0.4f, 0f), 1f / (1f + MathF.Exp(0.4f)));
            AssertNearly(ActivationsFunctions.Output(ActivationsFunctionType.Sigmoid, -0.8f, 0f), 1f / (1f + MathF.Exp(0.8f)));

            AssertNearly(ActivationsFunctions.Output(ActivationsFunctionType.Softsign, -0.5f, 0f), -0.33333333f);
            AssertNearly(ActivationsFunctions.Output(ActivationsFunctionType.Softsign, 0f, 0f), 0f);
            AssertNearly(ActivationsFunctions.Output(ActivationsFunctionType.Softsign, 0.5f, 0f), 0.33333333f);
            AssertNearly(ActivationsFunctions.Output(ActivationsFunctionType.Softsign, 0.8f, 0f), 0.44444444f);

            AssertNearly(ActivationsFunctions.Derivative(ActivationsFunctionType.Softsign, 0f, 0f), 1f);
            AssertNearly(ActivationsFunctions.Derivative(ActivationsFunctionType.Softsign, 0.2f, 0f), 0.69444444f);
            AssertNearly(ActivationsFunctions.Derivative(ActivationsFunctionType.Softsign, 0.5f, 0f), 0.44444444f);
            AssertNearly(ActivationsFunctions.Derivative(ActivationsFunctionType.Softsign, 0.8f, 0f), 0.30864198f);

            Console.WriteLine("pass");
        }

        private static void AssertNearly(float a, float b)
        {
            if (MathF.Abs(a - b) > 0.0000001f)
            {
                throw new Exception("a b is not equal");
            }
        }
    }
}
#endif