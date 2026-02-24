using Microsoft.AspNetCore.Mvc;
using SharpTorch.Runtime;
using System.Text;
using TorchSharp.Shared.Torch;

namespace SharpTorch.Api.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class TorchController : ControllerBase
    {
        private TorchService Service;

        public TorchController(TorchService service)
        {
            this.Service = service;
        }



        [HttpGet("status")]
        public ActionResult<TorchStatus> GetStatus()
        {
            try
            {
                var status = new TorchStatus
                {
                    DeviceId = this.Service.DeviceId,
                    DeviceName = this.Service.DeviceNameStr,
                    DeviceType = this.Service.DeviceTypeStr,
                    CudaEnabled = this.Service.IsCuda,
                    Initialized = true,
                    ActiveModel = this.Service.ActiveModelDto
                };

                return this.Ok(status);
            }
            catch (Exception ex)
            {
                return this.StatusCode(500, $"An error occurred while retrieving the torch status: {ex.Message}");
            }
        }

        [HttpGet("devices")]
        public async Task<ActionResult<int>> GetDevicesAsync()
        {
            try
            {
                int deviceCount = await this.Service.ListDevicesAsync();
                return this.Ok(deviceCount);
            }
            catch (Exception ex)
            {
                return this.StatusCode(500, $"An error occurred while retrieving the torch devices: {ex.Message}");
            }
        }


        [HttpGet("models")]
        public ActionResult<List<TorchModel>> GetModels()
        {
            try
            {
                var models = this.Service.AvailableModels;
                return this.Ok(models);
            }
            catch (Exception ex)
            {
                return this.StatusCode(500, $"An error occurred while retrieving the torch models: {ex.Message}");
            }
        }



        [HttpPost("start-model")]
        public async Task<ActionResult<string?>> StartModelAsync([FromBody] TorchModel? selectedModel = null)
        {
            if (selectedModel == null)
            {
                var models = this.Service.GetModels();
                selectedModel = models.FirstOrDefault();
                if (selectedModel == null)
                {
                    return this.NotFound("No available models to start.");
                }
            }

            try
            {
                await this.Service.StartModelAsync(selectedModel);
                return this.Ok(selectedModel.ModelRootPath);
            }
            catch (Exception ex)
            {
                return this.StatusCode(500, $"An error occurred while starting the model: {ex.Message}");
            }
        }

        [HttpDelete("unload")]
        public async Task<ActionResult<bool?>>UnloadModelAsync()
        {
            try
            {
                var result = await this.Service.UnloadModelAsync();
                if (result == null)
                {
                    return this.NotFound();
                }

                return this.Ok(result);
            }
            catch (Exception ex)
            {
                return this.StatusCode(500, $"An error occurred while unloading the model: {ex.Message}");
            }
        }



        // Streaming endpoints
        [HttpPost("generate/stream")]
        [Produces("text/event-stream")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status500InternalServerError)]
        public async Task<IActionResult> GenerateStreamAsync([FromBody] string? prompt = null, [FromQuery] int maxTokens = 512, CancellationToken ct = default)
        {
            prompt ??= "Hello this is a simple test to check if the LLM (you) gets this request and can respond. Please respond with 'LLM: OK' if you are alright and running.";

            if (this.Service.ActiveModel == null)
            {
                return this.BadRequest("No active model. Please start a model before generating text.");
            }

            var stream = this.Service.GenerateTextStreamAsync(prompt, maxTokens, ct);
            if (stream == null)
            {
                this.Response.StatusCode = StatusCodes.Status500InternalServerError;
                await this.Response.WriteAsync("data: Failed to start generation. Is a model loaded?\n\n", ct);
                return new EmptyResult();
            }

            this.Response.Headers.ContentType = "text/event-stream";
            this.Response.Headers.CacheControl = "no-cache";

            var responseBuilder = new StringBuilder();
            try
            {
                await foreach (var chunk in stream.WithCancellation(ct).ConfigureAwait(false))
                {
                    if (string.IsNullOrEmpty(chunk))
                    {
                        continue;
                    }

                    responseBuilder.Append(chunk);
                    await this.Response.WriteAsync($"data: {chunk}\n\n", ct);
                    await this.Response.Body.FlushAsync(ct);
                }
            }
            catch (OperationCanceledException)
            {
                if (!this.Response.HasStarted)
                {
                    return this.Ok(responseBuilder.ToString());
                }

                if (responseBuilder.Length > 0)
                {
                    await this.Response.WriteAsync($"data: {responseBuilder}\n\n", CancellationToken.None);
                    await this.Response.Body.FlushAsync(CancellationToken.None);
                }
            }

            return new EmptyResult();
        }

    }
}
