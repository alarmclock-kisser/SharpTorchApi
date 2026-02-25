using Radzen;
using Microsoft.AspNetCore.Components.Forms;
using SharpTorch.Client;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using TorchSharp.Shared.Torch;
using System.Runtime.CompilerServices;

namespace SharpTorch.WebApp.ViewModels
{
    public class MainViewModel
    {
        private readonly ApiClient Api;

        public MainViewModel(ApiClient api)
        {
            this.Api = api;
        }

        public TorchStatus? Status { get; private set; }
        public TorchModel? ActiveModel => this.Status?.ActiveModel;

        public ICollection<TorchModel> Models { get; private set; } = new List<TorchModel>();
        // bind to the selected model name (string) in the UI
        public string? SelectedModel { get; set; }

        public int? DeviceCount { get; private set; }
        public string? DeviceName { get; private set; }

        public string LoadButtonText => this.ActiveModel != null ? $"Unload" : "Load";
        public bool IsLoadButtonDisabled => this.ActiveModel == null && string.IsNullOrEmpty(this.SelectedModel) && (this.Models == null || !this.Models.Any());

        public async Task InitializeAsync()
        {
            await this.RefreshAsync();
        }

        public async Task RefreshAsync()
        {
            this.DeviceCount = await this.Api.GetDeviceCountAsync();

            this.Status = await this.Api.GetStatusAsync();
            if (this.Status != null)
            {
                this.DeviceName = this.Status.DeviceName;
                var models = await this.Api.GetModelsAsync();
                if (models != null)
                {
                    this.Models = models;
                }
            }
        }

        public async Task<string?> LoadModelAsync(string? modelName = null)
        {
            if (this.ActiveModel != null)
            {
                string unloadModel = this.ActiveModel.ModelName;
                // unload the current model first
                var unloadResult = await this.UnloadModelAsync();
                return $"Model '{unloadModel}' was unloaded. Unload result: {unloadResult}.";
            }

            var modelToLoad = modelName ?? this.SelectedModel ?? this.Models?.FirstOrDefault()?.ModelName;
            if (string.IsNullOrEmpty(modelToLoad))
            {
                return null;
            }

            var result = await this.Api.LoadModelAsync(modelToLoad);
            await this.RefreshAsync();
            return result;
        }

        public async Task<bool?> UnloadModelAsync()
        {
            var res = await this.Api.UnloadModelAsync();
            await this.RefreshAsync();
            return res;
        }

        public async IAsyncEnumerable<string> GenerateStreamAsync(string? prompt = null, int maxTokens = 512, [EnumeratorCancellation] CancellationToken ct = default)
        {
            await foreach (var chunk in this.Api.GenerateStreamAsync(prompt, maxTokens, ct))
            {
                yield return chunk;
            }
        }
    }
}
