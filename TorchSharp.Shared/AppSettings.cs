using System;
using System.Collections.Generic;
using System.Text;

namespace TorchSharp.Shared
{
    public class AppSettings
    {

        public string[] ModelDirectories { get; set; } = [];
        public bool ForceCpu { get; set; } = false;


    }
}
