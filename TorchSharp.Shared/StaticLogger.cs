using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.Serialization;
using System.Text;
using System.Threading;

namespace TorchSharp.Shared
{
    public static class StaticLogger
    {
        public static readonly ConcurrentDictionary<DateTime, string> LogEntries = new();
        public static readonly BindingList<string> LogEntriesBindingList = [];
        public static readonly BindingList<string> NativeLlamaLogEntriesBindingList = [];

        public static event Action<string>? LogAdded;

        public static string LogDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Logs");
        public static string? LogFilePath { get; private set; } = null;

        // UI synchronization context (set from the UI at startup)
        private static SynchronizationContext? UiContext;

        public static void SetUiContext(SynchronizationContext context)
        {
            UiContext = context;
        }

        public static void InitializeLogFiles(string? logDirectory = null, bool createLogFile = false, bool clearLogs = false)
        {
            LogDirectory = logDirectory ?? LogDirectory;

            try
            {
                if (!Directory.Exists(LogDirectory))
                {
                    Directory.CreateDirectory(LogDirectory);
                }

                if (clearLogs)
                {
                    Directory.Delete(LogDirectory, true);
                    Directory.CreateDirectory(LogDirectory);
                }

                if (createLogFile)
                {
                    LogFilePath = Path.Combine(LogDirectory, $"log_{DateTime.Now:yyyyMMdd_HHmmss}.txt");
                    File.Create(LogFilePath).Dispose();
                    Log($"Log file created at {LogFilePath}");
                }
            }
            catch (Exception ex)
            {
                Log($"Error with log files initialization: {ex.Message}");
            }
        }


        public static void Log(string message)
        {
            DateTime timestamp = DateTime.Now;
            string logEntry = $"[{timestamp:HH:mm:ss}] {message}";
            LogEntries[timestamp] = logEntry;

            if (!logEntry.Contains("[Native Llama]", StringComparison.OrdinalIgnoreCase))
            {
                if (UiContext != null)
                {
                    UiContext.Post(_ => LogEntriesBindingList.Add(logEntry), null);
                }
                else
                {
                    // Fallback: add on current thread
                    lock (LogEntriesBindingList)
                    {
                        LogEntriesBindingList.Add(logEntry);
                    }
                }

                LogAdded?.Invoke(logEntry);
            }
            else
            {
                if (UiContext != null)
                {
                    UiContext.Post(_ => NativeLlamaLogEntriesBindingList.Add(logEntry), null);
                }
                else
                {
                    // Fallback: add on current thread
                    lock (NativeLlamaLogEntriesBindingList)
                    {
                        NativeLlamaLogEntriesBindingList.Add(logEntry);
                    }
                }
            }

            Console.WriteLine(logEntry);
            if (LogFilePath != null)
            {
                try
                {
                    File.AppendAllText(LogFilePath, logEntry + Environment.NewLine);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error writing to log file: {ex.Message}");
                }
            }
        }

        public static void Log(Exception ex)
        {
            Log($"Exception: {ex.Message}\nStack Trace: {ex.StackTrace}");
        }

        public static async Task LogAsync(string message)
        {
            await Task.Run(() => Log(message));
        }

        public static async Task LogAsync(Exception ex)
        {
            await Task.Run(() => Log(ex));
        }



        public static void ClearLogs()
        {
            LogEntries.Clear();
            if (UiContext != null)
            {
                UiContext.Post(_ => LogEntriesBindingList.Clear(), null);
            }
            else
            {
                lock (LogEntriesBindingList)
                {
                    LogEntriesBindingList.Clear();
                }
            }
        }



    }
}