using System;
using System.Collections.Generic;
using System.Text;

namespace TorchSharp.Shared.Torch
{
    public class TorchModel
    {
        public string ModelRootPath { get; set; } = string.Empty;
        public string ModelName => Path.GetFileName(this.ModelRootPath);
        public string ModelPath { get; set; } = string.Empty;
        public string[] JsonFiles { get; set; } = Array.Empty<string>();



        // Parameterloser Konstruktor für Deserialisierung
        public TorchModel() { }

        public TorchModel(string? rootDir)
        {
            if (string.IsNullOrWhiteSpace(rootDir) || !Directory.Exists(rootDir))
            {
                Console.WriteLine($"Model directory not found: {rootDir}");
                this.ModelRootPath = string.Empty;
                this.ModelPath = string.Empty;
                this.JsonFiles = Array.Empty<string>();
                return;
            }

            this.ModelRootPath = rootDir;
            // IO auf sichere Weise ausführen
            this.ModelPath = Directory.GetFiles(this.ModelRootPath, "model.safetensors", SearchOption.AllDirectories)
                                      .FirstOrDefault() ?? string.Empty;
            this.JsonFiles = Directory.GetFiles(this.ModelRootPath, "*.json", SearchOption.AllDirectories);
        }
        


    }
}
