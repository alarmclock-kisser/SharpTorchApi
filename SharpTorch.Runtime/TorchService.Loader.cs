using SharpTorch.Runtime.Modules;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Text.Json;
using Tokenizers.HuggingFace;
using Tokenizers.HuggingFace.Tokenizer;
using TorchSharp;
using TorchSharp.PyBridge;
using TorchSharp.Shared;
using TorchSharp.Shared.Torch;
using static TorchSharp.torch;

namespace SharpTorch.Runtime
{
    public partial class TorchService
    {
        private Tokenizer? _tokenizer;


        public JsonDocument? ModelConfig { get; private set; }
        public JsonDocument? GenerationConfig { get; private set; }
        public JsonDocument? ChatTemplate { get; private set; }

        public nn.Module<Tensor, Tensor>? ActiveModel { get; private set; } = null;
        public TorchModel? ActiveModelDto { get; private set; } = null;


        public async Task StartModelAsync(TorchModel selectedModel)
        {
            await StaticLogger.LogAsync($"TorchService: Booting up model {selectedModel.ModelName}...");

            // 1. Lade alle relevanten JSON Configs
            await this.LoadConfigsFromListAsync(selectedModel.JsonFiles);

            // 2. Tokenizer initialisieren (sucht den Pfad aus der JsonFiles Liste)
            string? tokenizerPath = selectedModel.JsonFiles.FirstOrDefault(f => f.EndsWith("tokenizer.json"));
            if (!string.IsNullOrEmpty(tokenizerPath))
            {
                await this.InitializeTokenizerAsync(tokenizerPath);
            }

            // 3. Architektur Parameter aus der config.json extrahieren
            int vocabSize = 151936; // Standard Qwen3 Vocab Size Fallback
            int hiddenSize = 2048;  // Standard 2B Parameter Fallback
            int intermediateSize = 5632; // <-- NEU: Fallback für das MLP
            int numHeads = 16;       // <-- NEU
            int numKvHeads = 2;      // <-- NEU (GQA)
            int numLayers = 28;     // <-- NEU (Anz. Transformer-Blöcke)

            if (this.ModelConfig != null)
            {
                var root = this.ModelConfig.RootElement;
                if (root.TryGetProperty("vocab_size", out var v)) vocabSize = v.GetInt32();
                if (root.TryGetProperty("hidden_size", out var h)) hiddenSize = h.GetInt32();
                if (root.TryGetProperty("intermediate_size", out var i)) intermediateSize = i.GetInt32();
                if (root.TryGetProperty("num_attention_heads", out var nh)) numHeads = nh.GetInt32();
                if (root.TryGetProperty("num_key_value_heads", out var nk)) numKvHeads = nk.GetInt32();
                if (root.TryGetProperty("num_hidden_layers", out var nl)) numLayers = nl.GetInt32();
            }

            torch.set_default_dtype(ScalarType.BFloat16);

            // 4. Modell-Architektur in C# bauen
            this.ActiveModel = new Qwen3VLModel(vocabSize, hiddenSize, numHeads, numKvHeads, intermediateSize, numLayers);
            this.ActiveModelDto = selectedModel;

            // 5. Modell auf die GPU schieben und dann die 4GB Safetensors laden
            await this.LoadModelWeightsAsync(this.ActiveModel, selectedModel.ModelPath);

            await StaticLogger.LogAsync("TorchService: Model is fully loaded and ready for inference!");
        }

        private async Task LoadConfigsFromListAsync(IEnumerable<string> jsonFiles)
        {
            foreach (var path in jsonFiles)
            {
                if (!File.Exists(path))
                {
                    continue;
                }

                string fileName = Path.GetFileName(path);
                string json = await File.ReadAllTextAsync(path);

                switch (fileName)
                {
                    case "config.json":
                        this.ModelConfig = JsonDocument.Parse(json);
                        await StaticLogger.LogAsync("TorchService: Loaded config.json");
                        break;
                    case "generation_config.json":
                        this.GenerationConfig = JsonDocument.Parse(json);
                        await StaticLogger.LogAsync("TorchService: Loaded generation_config.json");
                        break;
                    case "chat_template.json":
                        this.ChatTemplate = JsonDocument.Parse(json);
                        await StaticLogger.LogAsync("TorchService: Loaded chat_template.json");
                        break;
                }
            }
        }


        public async Task LoadFromDirectoryAsync(string directoryPath, nn.Module model)
        {
            if (!Directory.Exists(directoryPath))
            {
                throw new DirectoryNotFoundException($"Path not found: {directoryPath}");
            }

            await StaticLogger.LogAsync($"TorchService: Scanning directory: {directoryPath}");

            // 1. Suche nach Safetensors (Hauptgewichte)
            string weightsPath = Path.Combine(directoryPath, "model.safetensors");
            if (!File.Exists(weightsPath))
            {
                // Falls gesharded: Suche nach dem ersten Teil oder Index
                weightsPath = Path.Combine(directoryPath, "model.safetensors.index.json");
            }

            // 2. Optionale Configs laden
            await this.LoadOptionalConfigsAsync(directoryPath);

            // 3. Tokenizer laden (falls vorhanden)
            string tokenizerPath = Path.Combine(directoryPath, "tokenizer.json");
            if (File.Exists(tokenizerPath))
            {
                await this.InitializeTokenizerAsync(tokenizerPath);
            }

            // 4. Gewichte in das Modell laden (falls das Modell bereits instanziiert wurde)
            if (model != null && File.Exists(weightsPath))
            {
                await this.LoadModelWeightsAsync(model, weightsPath);
            }
            else if (model == null)
            {
                await StaticLogger.LogAsync("TorchService: No module provided. Weights were not loaded into a model yet.");
            }
        }

        public async Task InitializeTokenizerAsync(string tokenizerJsonPath)
        {
            try
            {
                await StaticLogger.LogAsync($"TorchService: Loading tokenizer from {tokenizerJsonPath}...");

                // Wir laden den HuggingFace Tokenizer direkt in C#
                this._tokenizer = Tokenizers.HuggingFace.Tokenizer.Tokenizer.FromFile(tokenizerJsonPath);

                await StaticLogger.LogAsync("TorchService: Tokenizer loaded successfully.");
            }
            catch (Exception ex)
            {
                await StaticLogger.LogAsync(ex);
            }
        }

        public async Task<Tensor> EncodeTextAsync(string prompt)
        {
            await StaticLogger.LogAsync($"TorchService: Encoding prompt: {prompt}");

            // Die HuggingFace-C# Tokenizer-API liefert ein Encoding-Objekt.
            // Verwende direkt dessen Ids (RepeatedField<uint>) und konvertiere
            // sie zu einem long[] bevor wir torch.tensor aufrufen.
            var encoding = this._tokenizer.Encode(prompt, false);
            var uintIds = encoding.SelectMany(e => e.Ids).ToArray();
            var longIds = uintIds.Select(i => (long)i).ToArray();

            // Erzeuge einen Long-Tensor für das Modell (Batch-Size 1)
            var tensor = torch.tensor(longIds, dtype: ScalarType.Int64, device: this._device).unsqueeze(0);

            return tensor;
        }

        private async Task LoadOptionalConfigsAsync(string dir)
        {
            var files = new Dictionary<string, Action<JsonDocument>>
            {
                { "config.json", doc => this.ModelConfig = doc },
                { "generation_config.json", doc => this.GenerationConfig = doc }
            };

            foreach (var (fileName, assign) in files)
            {
                string path = Path.Combine(dir, fileName);
                if (File.Exists(path))
                {
                    await StaticLogger.LogAsync($"TorchService: Found and loading {fileName}");
                    string json = await File.ReadAllTextAsync(path);
                    assign(JsonDocument.Parse(json));
                }
            }
        }

        public async Task LoadModelWeightsAsync(nn.Module module, string filePath)
        {
            try
            {
                await StaticLogger.LogAsync($"TorchService: Loading Safetensors from {filePath}...");

                // GANZ WICHTIG: Zuerst auf Device schieben, damit Puffer in der GPU erstellt werden
                module.to(this._device);

                // 'strict: false' ignoriert Dummy-Layer Warnungen, lädt aber alles was passt
                module.load_safetensors(filePath, strict: false);

                // Nochmal syncen
                module.to(this._device);
                await StaticLogger.LogAsync($"TorchService: Weights loaded and module anchored to {this._device}.");
            }
            catch (Exception ex)
            {
                await StaticLogger.LogAsync($"TorchService ERROR: Could not load weights.");
                await StaticLogger.LogAsync(ex);
                throw;
            }
        }


    }
}
