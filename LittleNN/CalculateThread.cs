using System;
using System.Threading;

namespace LittleNN
{
    /// <summary>
    /// Logical calculation load thread
    /// </summary>
    public class CalculateThread
    {
        public const int AmountOfComputation = 5000 * 10;

        private static object m_Lock = new object();
        private static CalculateThread m_IdleThread;

        internal static CalculateThread IdleThread
        {
            get
            {
                lock (m_Lock)
                {
                    CalculateThread value;
                    if (m_IdleThread != null)
                    {
                        value = m_IdleThread;
                        m_IdleThread = null;
                    }
                    else
                        value = new CalculateThread();
                    value.Rent();
                    return value;
                }
            }
            set
            {
                lock (m_Lock)
                {
                    if (value.m_IsAbort)
                        throw new Exception("CalculateThread is abort");
                    if (m_IdleThread is null)
                    {
                        m_IdleThread = value;
                        m_IdleThread.Revert();
                    }
                    else
                        value.Abort();
                }
            }
        }

        private readonly AutoResetEvent m_WorkloadSignalEvent;
        private volatile bool m_Workload;
        /// <summary>
        /// Calculate thread is workload or idle
        /// </summary>
        public bool Workload => m_Workload;
        private Thread m_WorkThread;
        /// <summary>
        /// Return calculate thread state
        /// </summary>
        public ThreadState ThreadState => m_WorkThread.ThreadState;
        private volatile bool m_IsAbort;
        /// <summary>
        /// Calculate thread is abort?
        /// </summary>
        public bool IsAbort => m_IsAbort;
        private volatile Exception m_InnerException;
        /// <summary>
        /// If calculate thread have throw exception in task, there will store the exception.
        /// </summary>
        public Exception InnerException => m_InnerException;

        private readonly AutoResetEvent m_WorkFinishSignalEvent;
        private volatile Action m_Task;

        internal CalculateThread()
        {
            m_WorkloadSignalEvent = new AutoResetEvent(false);
            m_Workload = false;
            m_WorkThread = new Thread(ThreadUpdata);
            m_WorkThread.IsBackground = true;
            m_IsAbort = false;
            m_InnerException = null;

            m_WorkFinishSignalEvent = new AutoResetEvent(false);
            m_Task = null;

            m_WorkThread.Start();
        }

        private void Rent()
        {
            m_Workload = true;
            m_WorkloadSignalEvent.Reset();
            m_WorkFinishSignalEvent.Reset();
        }

        private void Revert()
        {
            m_Workload = false;
            m_WorkloadSignalEvent.Reset();
            m_WorkFinishSignalEvent.Reset();
            m_Task = null;
        }

        private void ThreadUpdata()
        {
            while (true)
            {
                bool haveTask = false;
                try
                {
                    m_WorkloadSignalEvent.WaitOne();
                    if (m_IsAbort)
                        break;
                    // Assume that tasks will not be given again when they already exist
                    // so i can get and set m_Task without lock
                    haveTask = true;
                    m_Task();
                    m_Task = null;
                }
                catch (Exception ex)
                {
                    Abort();
                    m_InnerException = ex;
                }
                finally
                {
                    if (haveTask)
                    {
                        m_WorkFinishSignalEvent.Set();
                    }
                }
            }

            m_WorkloadSignalEvent.Dispose();
            m_WorkFinishSignalEvent.Dispose();
        }

        internal void SetTask(Action action)
        {
            m_Task = action;
            m_WorkloadSignalEvent.Set();
        }
        /// <summary>
        /// Wait calculate thread finish task or throw exception
        /// <para>Current thread will be block before calculate thread finish task or throw exception</para>
        /// </summary>
        internal void WaitUntilFinish()
        {
            if (InnerException != null)
                throw InnerException;
            m_WorkFinishSignalEvent.WaitOne();
            if (InnerException != null)
                throw InnerException;
        }

        /// <summary>
        /// Set thread is abort, and expect it run to exit
        /// </summary>
        internal void Abort()
        {
            m_IsAbort = true;
            m_WorkloadSignalEvent.Set();
        }
    }
}