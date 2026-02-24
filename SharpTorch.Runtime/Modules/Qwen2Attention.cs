using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SharpTorch.Runtime.Modules
{
    public class Qwen2Attention : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> q_proj;
        private readonly Module<Tensor, Tensor> k_proj;
        private readonly Module<Tensor, Tensor> v_proj;
        private readonly Module<Tensor, Tensor> o_proj;

        private readonly Qwen2RotaryEmbedding rotary_emb; // <-- NEU

        private readonly int num_heads;
        private readonly int num_kv_heads;
        private readonly int head_dim;

        public Qwen2Attention(int hiddenSize, int numHeads, int numKvHeads) : base("Qwen2Attention")
        {
            this.num_heads = numHeads;
            this.num_kv_heads = numKvHeads;
            this.head_dim = hiddenSize / numHeads;

            this.q_proj = Linear(hiddenSize, numHeads * this.head_dim, hasBias: true);
            this.k_proj = Linear(hiddenSize, numKvHeads * this.head_dim, hasBias: true);
            this.v_proj = Linear(hiddenSize, numKvHeads * this.head_dim, hasBias: true);
            this.o_proj = Linear(numHeads * this.head_dim, hiddenSize, hasBias: false);

            this.rotary_emb = new Qwen2RotaryEmbedding(this.head_dim); // <-- NEU

            this.RegisterComponents();
        }

        // Hilfsmethode für die Rotation der Vektoren
        private Tensor ApplyRotaryPosEmb(Tensor x, Tensor cos, Tensor sin)
        {
            // x shape: [batch, num_heads, seq_len, head_dim]
            var chunks = x.chunk(2, dim: -1);
            var x1 = chunks[0];
            var x2 = chunks[1];

            var rotated_x = torch.cat(new[] { -x2, x1 }, dim: -1);

            // Dimensionen anpassen für Broadcasting
            cos = cos.unsqueeze(0).unsqueeze(0);
            sin = sin.unsqueeze(0).unsqueeze(0);

            return (x * cos) + (rotated_x * sin);
        }

        public override Tensor forward(Tensor x)
        {
            var batch = x.shape[0];
            var seq_len = x.shape[1];

            var q = this.q_proj.forward(x).view(batch, seq_len, this.num_heads, this.head_dim).transpose(1, 2);
            var k = this.k_proj.forward(x).view(batch, seq_len, this.num_kv_heads, this.head_dim).transpose(1, 2);
            var v = this.v_proj.forward(x).view(batch, seq_len, this.num_kv_heads, this.head_dim).transpose(1, 2);

            // --- NEU: RoPE (Positions-Wissen) anwenden! ---
            var (cos, sin) = this.rotary_emb.forward(seq_len, x.device);
            q = ApplyRotaryPosEmb(q, cos, sin);
            k = ApplyRotaryPosEmb(k, cos, sin);
            cos.Dispose();
            sin.Dispose();
            // ----------------------------------------------

            int num_key_value_groups = this.num_heads / this.num_kv_heads;
            if (num_key_value_groups > 1)
            {
                k = torch.repeat_interleave(k, num_key_value_groups, dim: 1);
                v = torch.repeat_interleave(v, num_key_value_groups, dim: 1);
            }

            var attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_casual: true);

            attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, this.num_heads * this.head_dim);
            return this.o_proj.forward(attn_output);
        }
    }
}