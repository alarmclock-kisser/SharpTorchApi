using System;
using System.Collections.Generic;
using System.Text;

namespace TorchSharp.Shared.Torch
{
    public class TorchStatus
    {
        public bool Initialized { get; set; } = false;
        public bool CudaEnabled { get; set; } = false;
        public int DeviceId { get; set; } = -1;
        public string DeviceName { get; set; } = string.Empty;
        public string DeviceType { get; set; } = string.Empty;

        public TorchModel? ActiveModel { get; set; } = null;


    }
}
