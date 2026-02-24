using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SharpTorch.Runtime.Modules
{
    // This is your nn.Module
    public class MultimodalProjector : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> _linear1;
        private readonly Module<Tensor, Tensor> _gelu;
        private readonly Module<Tensor, Tensor> _linear2;

        public MultimodalProjector(int inputDim, int outputDim) : base("Projector")
        {
            // Define layers - These names must match the keys in your .safetensors file!
            this._linear1 = Linear(inputDim, outputDim);
            this._gelu = GELU();
            this._linear2 = Linear(outputDim, outputDim);

            // Register all modules so .to(device) and .load_safetensors() can find them
            this.RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            x = this._linear1.forward(x);
            x = this._gelu.forward(x);
            return this._linear2.forward(x);
        }
    }
}