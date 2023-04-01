using System;

namespace LittleNN
{
    /// <summary>
    /// https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/#loss-functions
    /// copy from website
    /// </summary>
    public static class LossFuntion
    {
        /// <summary>
        /// equal torch.nn.L1Loss
        /// </summary>
        /// <param name="eval">NN forward return's array</param>
        /// <param name="target">the best return's array</param>
        public static float L1Loss(float[] eval, float[] target)
        {
            int evalLength = eval.Length;
            if (evalLength != target.Length)
                throw new ArgumentException($"rank is not equal, eval: {evalLength}, target: {target.Length}");
            if (evalLength == 0)
                throw new ArgumentException(nameof(eval));

            float loss = 0f;
            for (int i = 0; i < evalLength; i++)
            {
                float delta = eval[i] - target[i];
                if (delta > 0f)
                    loss += delta;
                else
                    loss += -delta;
            }
            return loss / evalLength;
        }
        /// <summary>
        /// equal torch.nn.MSELoss
        /// </summary>
        /// <param name="eval">NN forward return's array</param>
        /// <param name="target">the best return's array</param>
        public static float MSELoss(float[] eval, float[] target)
        {
            int evalLength = eval.Length;
            if (evalLength != target.Length)
                throw new ArgumentException($"rank is not equal, eval: {evalLength}, target: {target.Length}");
            if (evalLength == 0)
                throw new ArgumentException(nameof(eval));

            float loss = 0f;
            for (int i = 0; i < evalLength; i++)
            {
                float delta = eval[i] - target[i];
                loss += delta * delta;
            }
            return loss / evalLength;
        }
    }
}