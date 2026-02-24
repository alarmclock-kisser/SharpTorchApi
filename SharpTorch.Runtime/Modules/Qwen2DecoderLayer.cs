using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SharpTorch.Runtime.Modules
{
    public class Qwen2DecoderLayer : Module<Tensor, Tensor>
    {
        private readonly Qwen2Attention self_attn;
        private readonly Qwen2MLP mlp;
        private readonly Qwen2RMSNorm input_layernorm;
        private readonly Qwen2RMSNorm post_attention_layernorm;

        public Qwen2DecoderLayer(int hiddenSize, int numHeads, int numKvHeads, int intermediateSize) : base("Qwen2DecoderLayer")
        {
            this.self_attn = new Qwen2Attention(hiddenSize, numHeads, numKvHeads);
            this.mlp = new Qwen2MLP(hiddenSize, intermediateSize);
            this.input_layernorm = new Qwen2RMSNorm(hiddenSize);
            this.post_attention_layernorm = new Qwen2RMSNorm(hiddenSize);

            this.RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            // Pfad 1: Attention mit Residual Connection
            var residual = x;
            x = this.input_layernorm.forward(x);
            x = this.self_attn.forward(x);
            x = residual + x; // Der berühmte Skip-Connection

            // Pfad 2: MLP mit Residual Connection
            residual = x;
            x = this.post_attention_layernorm.forward(x);
            x = this.mlp.forward(x);
            x = residual + x;

            return x;
        }
    }
}