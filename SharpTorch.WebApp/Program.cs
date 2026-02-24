using SharpTorch.Client;
using SharpTorch.WebApp.Components;
using SharpTorch.WebApp.ViewModels;

namespace SharpTorch.WebApp
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // CORS für API-Zugriff
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

            // HttpClient für API
            var apiBase = builder.Configuration.GetValue<string>("ApiBaseUrl") ?? "https://localhost:7224/";
            var timeout = builder.Configuration.GetValue<int?>("MaxTimeout") ?? 300;

            builder.Services.AddSingleton<ApiClient>(provider => new ApiClient(apiBase, timeout));

            // HTTPS-Umleitung und HSTS aktivieren
            builder.Services.AddHttpsRedirection(options =>
            {
                options.RedirectStatusCode = StatusCodes.Status308PermanentRedirect;
            });

            builder.Services.AddHsts(options =>
            {
                options.Preload = true;
                options.IncludeSubDomains = true;
                options.MaxAge = TimeSpan.FromDays(365);
            });

            // Antiforgery-Cookie für die Sicherstellung von SameSite-Attributen
            builder.Services.AddAntiforgery(options =>
            {
                options.Cookie.SameSite = SameSiteMode.None;
                options.Cookie.SecurePolicy = CookieSecurePolicy.Always;
                options.Cookie.HttpOnly = true;
                options.HeaderName = "X-CSRF-TOKEN";
            });

            builder.Services.AddRazorPages();
            builder.Services.AddServerSideBlazor();
            builder.Services.AddSignalR();
            // Radzen services for Blazor Server (dialogs, notifications, tooltip, context menus)
            builder.Services.AddScoped<Radzen.DialogService>();
            builder.Services.AddScoped<Radzen.NotificationService>();
            builder.Services.AddScoped<Radzen.TooltipService>();
            builder.Services.AddScoped<Radzen.ContextMenuService>();
            builder.Services.AddScoped<MainViewModel>();

            var app = builder.Build();

            // HTTP-Pipeline konfigurieren
            if (!app.Environment.IsDevelopment())
            {
                app.UseExceptionHandler("/Error");
            }

            app.UseHsts();
            app.UseHttpsRedirection();

            app.UseStaticFiles();

            app.UseRouting();

            app.UseCors(CorsPolicy);

            app.UseAntiforgery();

            // WebSockets für Blazor verwenden
            app.UseWebSockets();
            app.UseAuthorization();

            // Blazor Server-Endpunkte
            app.MapBlazorHub();
            app.MapFallbackToPage("/_Host");

            app.Run();
        }
    }
}
