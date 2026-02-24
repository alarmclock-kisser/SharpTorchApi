using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SharpTorch.Runtime.Modules
{
    // Die Standard-Normalisierung für Qwen-Modelle
    public class Qwen2RMSNorm : Module<Tensor, Tensor>
    {
        private readonly double _eps;
        private readonly Parameter weight;

        public Qwen2RMSNorm(int hiddenSize, double eps = 1e-6) : base("Qwen2RMSNorm")
        {
            this._eps = eps;
            // Erstellt die lernbaren Gewichte für diese Schicht, 
            // welche von den SafeTensors überschrieben werden
            this.weight = Parameter(torch.ones(new long[] { hiddenSize }));

            this.RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            // Da wir mit BFloat16 arbeiten, ist es oft sicherer, die Varianz in Float32 zu berechnen 
            // und danach wieder auf den Input-Typ (BFloat16) zurückzugehen
            var originalType = x.dtype;
            var xF32 = x.to_type(ScalarType.Float32);

            // 1. x quadrieren und den Durchschnitt über die letzte Dimension bilden
            var variance = xF32.pow(2).mean(new long[] { -1 }, keepdim: true);

            // 2. Normalisieren (1 / sqrt(variance + eps))
            var hidden_states = xF32 * torch.rsqrt(variance + this._eps);

            // 3. Typ zurückwandeln und mit den gelernten Gewichten multiplizieren
            return this.weight * hidden_states.to_type(originalType);
        }
    }
}