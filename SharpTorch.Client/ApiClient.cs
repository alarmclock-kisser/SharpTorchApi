using System;
using System.Collections.Generic;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text;
using TorchSharp.Shared.Torch;

namespace SharpTorch.Client
{
    public class ApiClient
    {
        private readonly HttpClient Http;
        private readonly InternalClient Client;

        public ApiClient(string baseUrl, int timeout = 300)
        {
            this.Http = new HttpClient
            {
                BaseAddress = new Uri(baseUrl),
                Timeout = TimeSpan.FromSeconds(timeout)
            };
            this.Client = new InternalClient(baseUrl, this.Http);
        }


        public async Task<TorchStatus?> GetStatusAsync()
        {
            try
            {
                return await this.Client.StatusAsync();

            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error getting status: {ex.Message}");
                return null;
            }
        }

        public async Task<int?> GetDeviceCountAsync()
        {
            try
            {
                return await this.Client.DevicesAsync();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error getting device count: {ex.Message}");
                return null;
            }
        }

        public async Task<ICollection<TorchModel>> GetModelsAsync()
        {
            try
            {
                return await this.Client.ModelsAsync();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error getting models: {ex.Message}");
                return [];
            }
        }

        public async Task<string?> LoadModelAsync(string modelName)
        {
            try
            {
                var models = await this.GetModelsAsync();
                var model = models.FirstOrDefault(m => m.ModelName.Equals(modelName, StringComparison.OrdinalIgnoreCase));
                if (model == null)
                {
                    Console.WriteLine("Model not found: " + modelName);
                    return null;
                }

                string result = await this.Client.StartModelAsync(model) ?? "Failed to start model for unknown reasons.";
                return result;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading model '{modelName}': {ex.Message}");
                return $"Error loading model '{modelName}': {ex.Message}";
            }
        }

        public async Task<bool?> UnloadModelAsync()
        {
            try
            {
                var result = await this.Client.UnloadAsync();
                return result;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error unloading model: {ex.Message}");
                return null;
            }
        }

        // Llama SSE streaming generate
        public async IAsyncEnumerable<string> GenerateStreamAsync(string? prompt = null, int maxTokens = 512, [EnumeratorCancellation] CancellationToken ct = default)
        {
            // Create request and ensure we accept SSE
            // POST to API controller route and put maxTokens in query string (controller expects prompt from body and maxTokens from query)
            var requestUri = $"api/Torch/generate/stream?maxTokens={maxTokens}";
            using var request = new HttpRequestMessage(HttpMethod.Post, requestUri)
            {
                // Controller expects a string body for prompt; serialize prompt as JSON string
                Content = JsonContent.Create<string?>(prompt)
            };
            request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("text/event-stream"));

            HttpResponseMessage? response = null;
            try
            {
                response = await this.Http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);

                if (response == null)
                {
                    Console.WriteLine("No response received from server.");
                    yield break;
                }

                if (!response.IsSuccessStatusCode)
                {
                    // Read error body (if any) for diagnostics and don't throw from iterator
                    string? body = null;
                    try
                    {
                        body = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
                    }
                    catch { /* ignore read errors */ }

                    Console.WriteLine($"GenerateStreamAsync failed. Status: {(int)response.StatusCode} {response.ReasonPhrase}. Body: {body}");
                    response.Dispose();
                    yield break;
                }
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                // Cancellation requested - stop enumeration silently
                response?.Dispose();
                yield break;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error sending generate/stream request: {ex.Message}");
                response?.Dispose();
                yield break;
            }

            using (response)
            {
                await foreach (var item in ReadSseAsync(response, ct).ConfigureAwait(false))
                {
                    yield return item;
                }
            }
        }






        private static async IAsyncEnumerable<string> ReadSseAsync(HttpResponseMessage response, [EnumeratorCancellation] CancellationToken ct)
        {
            await using var stream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
            using var reader = new StreamReader(stream);
            var dataBuilder = new StringBuilder();


            string? line;
            while ((line = await reader.ReadLineAsync(ct).ConfigureAwait(false)) != null)
            {
                ct.ThrowIfCancellationRequested();

                if (line.Length == 0)
                {
                    if (dataBuilder.Length > 0)
                    {
                        yield return dataBuilder.ToString();
                        dataBuilder.Clear();
                    }
                    continue;
                }

                if (line.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
                {
                    var payload = line.Length > 5 ? line[5..] : string.Empty;
                    if (payload.StartsWith(" ", StringComparison.Ordinal))
                    {
                        payload = payload[1..];
                    }

                    dataBuilder.Append(payload);
                }
            }

            if (dataBuilder.Length > 0)
            {
                yield return dataBuilder.ToString();
            }
        }

    }
}
