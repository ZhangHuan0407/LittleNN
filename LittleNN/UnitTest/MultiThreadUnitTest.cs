#if DEBUG || UNIT_TEST
using System;
using System.Threading;
using LittleNN;

namespace UnitTest
{
    public static class MultiThreadUnitTest
    {
        public class CustomException : Exception
        {
        }

        public static void ThreadFunctionUnitTest()
        {
            Console.WriteLine("ThreadFunctionUnitTest");

            // get idle thread and run task
            CalculateThread firstCalculateThread = CalculateThread.IdleThread;
            if (!firstCalculateThread.Workload)
                throw new Exception("CalculateThread Workload 1");
            if (firstCalculateThread.IsAbort)
                throw new Exception("CalculateThread IsAbort 1");
            if (!firstCalculateThread.ThreadState.HasFlag(ThreadState.Background))
                throw new Exception("CalculateThread ThreadState dont have Background");
            for (int i = 0; i < 1000; i++)
            {
                firstCalculateThread.SetTask(() =>
                {
                });
                firstCalculateThread.WaitUntilFinish();
                if (!firstCalculateThread.ThreadState.HasFlag(ThreadState.WaitSleepJoin) &&
                    !firstCalculateThread.ThreadState.HasFlag(ThreadState.Running))
                    throw new Exception("CalculateThread ThreadState");
            }

            // test thread recycle
            CalculateThread secondCalculateThread = CalculateThread.IdleThread;
            CalculateThread.IdleThread = firstCalculateThread;
            if (firstCalculateThread.Workload)
                throw new Exception("CalculateThread Workload 2");
            if (firstCalculateThread.IsAbort)
                throw new Exception("CalculateThread IsAbort 2");
            if (secondCalculateThread == firstCalculateThread)
                throw new Exception("secondCalculateThread == firstCalculateThread");
            CalculateThread.IdleThread = secondCalculateThread;
            Thread.Sleep(2000);
            if (secondCalculateThread.ThreadState == ThreadState.Aborted || secondCalculateThread.ThreadState == ThreadState.AbortRequested)
                throw new Exception("CalculateThread thread is not abort");
            firstCalculateThread = null;
            CalculateThread firstThreadRef2 = CalculateThread.IdleThread;
            if (!firstThreadRef2.Workload)
                throw new Exception("CalculateThread Workload 3");
            if (firstThreadRef2.IsAbort)
                throw new Exception("CalculateThread IsAbort 3");

            // throw exception
            firstThreadRef2.SetTask(() =>
            {
                throw new CustomException();
            });
            bool haveThrowException = false;
            try
            {
                firstThreadRef2.WaitUntilFinish();
            }
            catch (Exception ex)
            {
                haveThrowException = ex is CustomException;
            }
            if (!haveThrowException)
            {
                throw new Exception("CalculateThread WaitUntilFinish miss exception");
            }
            if (!firstThreadRef2.IsAbort)
                throw new Exception("CalculateThread is not abort");

            // wait until must wait
            Console.WriteLine("test thread sync");
            CalculateThread thirdCalculateThread = CalculateThread.IdleThread;
            int value = 0;
            for (int i = 0; i < 2000; i++)
            {
                thirdCalculateThread.SetTask(() =>
                {
                    Thread.Sleep(1);
                    if (value != 0)
                        throw new Exception();
                    int p = --value;
                    if (p != -1)
                        throw new Exception();
                });
                thirdCalculateThread.WaitUntilFinish();
                value++;
            }
            Console.WriteLine("all pass");
        }
    }
}
#endif