using TorchSharp;
using static TorchSharp.torch;

namespace SharpTorch.Runtime.Modules
{
    // FIX: Wir erben nur vom Basis-Module (ohne <Tensor, Tensor>), 
    // um unsere eigene Signatur für die forward-Methode nutzen zu können.
    public class Qwen2RotaryEmbedding : torch.nn.Module
    {
        private readonly int dim;
        private readonly double base_val;

        public Qwen2RotaryEmbedding(int dim, double base_val = 1000000.0) : base("Qwen2RotaryEmbedding")
        {
            this.dim = dim;
            this.base_val = base_val;
        }

        // Kein "override" mehr, einfach eine normale öffentliche Methode
        public (Tensor cos, Tensor sin) forward(long seq_len, Device device)
        {
            using var scope = torch.NewDisposeScope();

            // Frequenz-Berechnung
            var inv_freq = 1.0 / torch.pow(this.base_val, torch.arange(0, this.dim, 2, dtype: ScalarType.Float32, device: device) / this.dim);
            var t = torch.arange(seq_len, dtype: ScalarType.Float32, device: device);

            var freqs = torch.outer(t, inv_freq);

            // Konkatenieren für die komplexe Rotation
            var emb = torch.cat(new[] { freqs, freqs }, dim: -1);

            // Wir brauchen Cosinus und Sinus in BFloat16
            var cos = emb.cos().to_type(ScalarType.BFloat16).MoveToOuterDisposeScope();
            var sin = emb.sin().to_type(ScalarType.BFloat16).MoveToOuterDisposeScope();

            return (cos, sin);
        }
    }
}