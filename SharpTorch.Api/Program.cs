using SharpTorch.Runtime;
using TorchSharp.Shared;
using Microsoft.OpenApi;

namespace SharpTorch.Api
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // Appsettings
            var configuration = new ConfigurationBuilder()
                .AddJsonFile("appsettings.json", optional: false, reloadOnChange: true)
                .Build();

            var appSettings = configuration.GetSection("AppSettings").Get<AppSettings>() ?? new AppSettings();

            // CORS policy
            const string CorsPolicy = "AllowApi";
            builder.Services.AddCors(options =>
            {
                options.AddPolicy(CorsPolicy, policy =>
                {
                    policy
                        .AllowAnyHeader()
                        .AllowAnyMethod()
                        .AllowCredentials()
                        .SetIsOriginAllowed(_ => true);
                });
            });

            // Add services to the container.
            builder.Services.AddSingleton(new TorchService(appSettings.ModelDirectories,appSettings.ForceCpu));
            StaticLogger.InitializeLogFiles(createLogFile: true);

            builder.Services.AddControllers();
            builder.Services.AddSignalR();
            builder.Services.AddEndpointsApiExplorer();
            builder.Services.AddSwaggerGen();

            var app = builder.Build();

            /// Configure the HTTP request pipeline.
            if (app.Environment.IsDevelopment())
            {
                app.UseSwagger();
                app.UseSwaggerUI();
            }

            app.UseHsts();
            app.UseHttpsRedirection();

            app.UseCors(CorsPolicy);

            app.UseAuthorization();

            app.MapControllers();

            app.Run();
        }
    }
}
