using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SharpTorch.Runtime.Modules
{
    // Die Backbone-Klasse (Das reine Text-Modell ohne Multimodal)
    public class Qwen2Model : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> embed_tokens;
        private readonly ModuleList<Qwen2DecoderLayer> layers;
        private readonly Qwen2RMSNorm norm;

        public Qwen2Model(int vocabSize, int hiddenSize, int numHeads, int numKvHeads, int intermediateSize, int numLayers) : base("model")
        {
            this.embed_tokens = Embedding(vocabSize, hiddenSize);

            this.layers = new ModuleList<Qwen2DecoderLayer>();
            for (int i = 0; i < numLayers; i++)
            {
                this.layers.append(new Qwen2DecoderLayer(hiddenSize, numHeads, numKvHeads, intermediateSize));
            }

            this.norm = new Qwen2RMSNorm(hiddenSize);
            this.RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            // 1. Tokens in Vektoren umwandeln (kommt leider oft als Float32 raus)
            x = this.embed_tokens.forward(x);

            // 2. DER FIX: Wir zwingen den Tensor unwiderruflich auf BFloat16!
            x = x.to_type(ScalarType.BFloat16);

            // 3. Ab durch die 28 Transformer-Layer
            foreach (var layer in this.layers)
            {
                x = layer.forward(x);
            }

            return this.norm.forward(x);
        }
    }

    // Die Hauptklasse (Wrapper)
    public class Qwen3VLModel : Module<Tensor, Tensor>
    {
        // Diese Variable MUSS 'model' heißen, weil die Gewichte in den Safetensors "model.layers..." heißen!
        private readonly Qwen2Model model;
        private readonly Module<Tensor, Tensor> lm_head;
        private readonly int _vocabSize;

        public Qwen3VLModel(int vocabSize, int hiddenSize, int numHeads, int numKvHeads, int intermediateSize, int numLayers) : base("Qwen3_VL")
        {
            this._vocabSize = vocabSize;

            // Text-Gehirn initialisieren
            this.model = new Qwen2Model(vocabSize, hiddenSize, numHeads, numKvHeads, intermediateSize, numLayers);

            this.lm_head = Linear(hiddenSize, vocabSize, hasBias: false);
            this.RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            x = x.to_type(ScalarType.Int64);
            x = torch.clamp(x, min: 0, max: this._vocabSize - 1);

            var features = this.model.forward(x);
            features = features.to_type(ScalarType.BFloat16);

            return this.lm_head.forward(features);
        }
    }
}