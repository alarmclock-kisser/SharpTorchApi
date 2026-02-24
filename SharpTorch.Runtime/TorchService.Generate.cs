using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Shared;
using static TorchSharp.torch;

namespace SharpTorch.Runtime
{
    public partial class TorchService
    {
        public async IAsyncEnumerable<string> GenerateTextStreamAsync(
            string prompt,
            int maxTokens = 512,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            if (this.ActiveModel == null)
            {
                throw new InvalidOperationException("Model is not loaded. Call StartModelAsync first.");
            }

            if (this._tokenizer == null)
            {
                throw new InvalidOperationException("Tokenizer is not initialized. Call StartModelAsync first.");
            }

            await StaticLogger.LogAsync("TorchService: Starting generation stream for prompt...");

            // 1. Prompt in Token-IDs umwandeln
            var inputIds = await EncodeTextAsync(prompt);

            // Qwen Stop-Token
            long eosTokenId = 151643;

            // NEU: Listen für sauberes Tokenizer-Decoding
            List<uint> generatedTokens = new List<uint>();
            string previousText = "";

            for (int i = 0; i < maxTokens; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                long nextTokenId;
                Tensor newInputIds;

                // Gradienten-Berechnung deaktivieren
                using (var noGrad = torch.no_grad())
                // Scope für Speichermanagement
                using (var scope = torch.NewDisposeScope())
                {
                    // Forward-Pass
                    var logits = this.ActiveModel.forward(inputIds);

                    // Letztes Token extrahieren
                    var lastTokenLogits = logits[0, -1];
                    var nextTokenTensor = lastTokenLogits.argmax();
                    nextTokenId = nextTokenTensor.item<long>();

                    // Neuen Input zusammenbauen
                    var nextTokenUnsq = torch.tensor(new[] { nextTokenId }, device: this._device).unsqueeze(0);
                    newInputIds = torch.cat(new[] { inputIds, nextTokenUnsq }, dim: 1);

                    // Memory Management
                    newInputIds.DetachFromDisposeScope();
                    inputIds.Dispose();
                    inputIds = newInputIds;
                }

                // Abbruchbedingung prüfen
                if (nextTokenId == eosTokenId)
                {
                    await StaticLogger.LogAsync("TorchService: EOS token reached. Generation finished.");
                    break;
                }

                // --- NEUER TOKENIZER DECODE BLOCK ---

                // 1. Token der Historie hinzufügen
                generatedTokens.Add((uint) nextTokenId);

                // 2. Den gesamten bisherigen Text dekodieren
                string currentText = this._tokenizer.Decode(generatedTokens.ToArray(), skipSpecialTokens: true);

                // 3. Nur die Differenz (das wirklich neue Stück Text) ermitteln
                if (currentText.Length > previousText.Length)
                {
                    string newTextChunk = currentText.Substring(previousText.Length);
                    previousText = currentText; // Update für den nächsten Durchlauf

                    // 4. Den sauberen Chunk streamen
                    yield return newTextChunk;
                }

                if (i % 2 == 0)
                {
                    GC.Collect();
                }

                await Task.Yield();
            }

            // Cleanup
            inputIds.Dispose();
        }
    }
}