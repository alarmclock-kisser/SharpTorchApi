using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SharpTorch.Runtime.Modules
{
    public class Qwen2MLP : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> gate_proj;
        private readonly Module<Tensor, Tensor> up_proj;
        private readonly Module<Tensor, Tensor> down_proj;
        private readonly Module<Tensor, Tensor> act_fn; // Die Aktivierungsfunktion (SiLU)

        public Qwen2MLP(int hiddenSize, int intermediateSize) : base("Qwen2MLP")
        {
            // Qwen nutzt hier keine Bias-Werte, nur reine Gewichte
            this.gate_proj = Linear(hiddenSize, intermediateSize, hasBias: false);
            this.up_proj = Linear(hiddenSize, intermediateSize, hasBias: false);
            this.down_proj = Linear(intermediateSize, hiddenSize, hasBias: false);

            // SiLU (Sigmoid Linear Unit) wird bei Qwen auch Swish genannt
            this.act_fn = SiLU();

            this.RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            // 1. Die Daten in zwei parallele Pfade schicken
            var gate = this.gate_proj.forward(x);
            var up = this.up_proj.forward(x);

            // 2. Den "Türsteher" (Gate) durch die SiLU-Funktion aktivieren
            var activated_gate = this.act_fn.forward(gate);

            // 3. Beide Pfade zusammenführen (Elementweise Multiplikation)
            var intermediate = activated_gate * up;

            // 4. Zurück auf die ursprüngliche Größe projizieren
            return this.down_proj.forward(intermediate);
        }
    }
}