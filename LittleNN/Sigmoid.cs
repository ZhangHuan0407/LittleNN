using System;

namespace LittleNN
{
    /// <summary>
    /// This is math function class
    /// </summary>
    public static class Sigmoid
    {
        /// <summary>
        ///  Math function of Sigmoid
        /// <para>1f / (1f + MathF.Exp(-x))</para>
        /// </summary>
        public static float Output(float x)
        {
            if (x < -45f)
                return 0f;
            else if (x > 45f)
                return 1f;
            else
                return 1f / (1f + MathF.Exp(-x));
        }

        /// <summary>
        /// Math function of Sigmoid's derivative
        /// <para>x * (1 - x)</para>
        /// </summary>
        public static float Derivative(float x) => x * (1 - x);
    }
}