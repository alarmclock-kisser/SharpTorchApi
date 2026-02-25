using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Shared;
using TorchSharp.Shared.Torch;
using static TorchSharp.torch;

namespace SharpTorch.Runtime
{
    public partial class TorchService : IDisposable
    {
        public List<string> ModelDirectories { get; set; } = [];
        public List<TorchModel> AvailableModels { get; set; } = [];


        public int DeviceId => this._device.index;
        public string DeviceNameStr => this._device.ToString();
        public string DeviceTypeStr => this._device.type.ToString();

        public bool IsCuda => this._device.type == DeviceType.CUDA;

        private readonly torch.Device _device;
        private bool _isDisposed = false;


        public TorchService(IEnumerable<string>? modelDirectories = null, bool forceCpu = false)
        {
            this.AvailableModels = this.GetModels(modelDirectories);

            try
            {
                // Check for CUDA availability
                if (!forceCpu && cuda.is_available())
                {
                    StaticLogger.Log("TorchService: CUDA is available. Initializing GPU mode.");
                    this._device = CUDA;
                }
                else
                {
                    StaticLogger.Log("TorchService: Initializing CPU mode.");
                    this._device = CPU;
                }
            }
            catch (Exception ex)
            {
                StaticLogger.Log(ex);
                this._device = CPU;
            }
        }


        public List<TorchModel> GetModels(IEnumerable<string>? modelDirectories = null)
        {
            if (modelDirectories != null)
            {
                this.ModelDirectories = modelDirectories.ToList();
            }

            var dirs = this.ModelDirectories.SelectMany(d => Directory.GetDirectories(d));

            var models = new List<TorchModel>();
            foreach (var dir in dirs.Where(d => Directory.GetFiles(d, "*.safetensors", SearchOption.AllDirectories).Any()))
            {
                try
                {
                    var model = new TorchModel(dir);
                    models.Add(model);
                }
                catch (Exception ex)
                {
                    StaticLogger.Log(ex);
                }
            }

            return models;
        }


        public async Task<int> ListDevicesAsync()
        {
            try
            {
                await StaticLogger.LogAsync("--- TorchSharp Device Information ---");
                await StaticLogger.LogAsync($"Selected Default Device: {this._device}");

                if (cuda.is_available())
                {
                    int deviceCount = cuda.device_count();
                    await StaticLogger.LogAsync($"CUDA Device Count: {deviceCount}");

                    for (int i = 0; i < deviceCount; i++)
                    {
                        // torch.cuda does not expose a Device type; create a torch.Device via torch.device
                        string name = torch.device("cuda", i).ToString();
                        await StaticLogger.LogAsync($"  [Device {i}]: {name}");
                    }

                    return deviceCount;
                }
                else
                {
                    await StaticLogger.LogAsync("No CUDA-compatible GPUs detected on this system.");

                    return 0;
                }
            }
            catch (Exception ex)
            {
                await StaticLogger.LogAsync(ex);
                return 0;
            }
            finally
            {
                await StaticLogger.LogAsync("--- End of Device Information ---");
            }
        }

        public Dictionary<int, string> GetDeviceTypes()
        {
            var deviceTypes = new Dictionary<int, string>();

            if (cuda.is_available())
            {
                int deviceCount = cuda.device_count();
                for (int i = 0; i < deviceCount; i++)
                {
                    string name = torch.device("cuda", i).ToString();
                    deviceTypes.Add(i, name);
                }
            }

            return deviceTypes;
        }


        public async Task<bool?> UnloadModelAsync()
        {
            if (this.ActiveModel == null)
            {
                await StaticLogger.LogAsync("TorchService: No active model to unload.");
                return null;
            }
            try
            {
                await StaticLogger.LogAsync("TorchService: Unloading active model (aggressive cleanup)...");

                // 1) Try to dispose tensors that belong to the model (parameters + buffers)
                try
                {
                    var model = this.ActiveModel;
                    if (model != null)
                    {
                        try
                        {
                            // Dispose parameters
                            var pars = model.parameters();
                            if (pars != null)
                            {
                                foreach (var p in pars)
                                {
                                    try { p.Dispose(); } catch { }
                                }
                            }
                        }
                        catch (Exception ex)
                        {
                            StaticLogger.Log(ex);
                        }

                        try
                        {
                            // Dispose buffers
                            var bufs = model.buffers();
                            if (bufs != null)
                            {
                                foreach (var b in bufs)
                                {
                                    try { b.Dispose(); } catch { }
                                }
                            }
                        }
                        catch (Exception ex)
                        {
                            StaticLogger.Log(ex);
                        }

                        // Finally dispose the module itself
                        try { model.Dispose(); } catch (Exception ex) { StaticLogger.Log(ex); }
                    }
                }
                catch (Exception ex)
                {
                    // Log and continue with best-effort cleanup
                    await StaticLogger.LogAsync(ex);
                }

                // 2) Clear references to DTOs/configs/tokenizer so they become collectible
                this.ActiveModel = null;
                this.ActiveModelDto = null;

                // Tokenizer may not implement IDisposable in all builds — just clear the reference
                this._tokenizer = null;

                try { this.ModelConfig?.Dispose(); } catch { }
                this.ModelConfig = null;
                try { this.GenerationConfig?.Dispose(); } catch { }
                this.GenerationConfig = null;
                try { this.ChatTemplate?.Dispose(); } catch { }
                this.ChatTemplate = null;

                // 3) If CUDA is available, try to synchronize and clear caches
                try
                {
                    if (cuda.is_available())
                    {
                        cuda.synchronize();
                        await StaticLogger.LogAsync("TorchService: CUDA synchronized.");
                    }
                }
                catch (Exception ex)
                {
                    await StaticLogger.LogAsync(ex);
                }

                // 4) Force multiple GC cycles and wait for finalizers to run
                for (int i = 0; i < 3; i++)
                {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    await Task.Delay(50);
                }

                await StaticLogger.LogAsync("TorchService: Model unloaded successfully.");
                return true;
            }
            catch (Exception ex)
            {
                await StaticLogger.LogAsync(ex);
                return false;
            }
        }


        public void Dispose()
        {
            if (!this._isDisposed)
            {
                // No unmanaged resources to clean up, but if there were, we'd do it here.
                this._isDisposed = true;
            }

            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!this._isDisposed)
            {
                if (disposing)
                {
                    StaticLogger.Log("TorchService: Disposing managed resources.");
                }

                try
                {
                    if (cuda.is_available())
                    {
                        // Synchronize the CUDA device
                        cuda.synchronize();
                        StaticLogger.Log("TorchService: CUDA device synchronized successfully.");
                    }
                }
                catch (Exception ex)
                {
                    StaticLogger.Log(ex);
                }

                this._isDisposed = true;
                StaticLogger.Log("TorchService: Shutdown complete.");
            }
        }


    }
}
