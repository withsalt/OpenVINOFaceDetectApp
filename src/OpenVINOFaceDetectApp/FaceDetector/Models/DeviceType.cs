using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OpenVINOFaceDetectApp.FaceDetector.Models
{
    public enum DeviceType
    {
        [Description("CPU")]
        CPU,

        [Description("GPU")]
        GPU,

        [Description("GPU.0")]
        GPU_0,

        [Description("GPU.1")]
        GPU_1,

        [Description("GPU.2")]
        GPU_2,

        [Description("NPU")]
        NPU
    }
}
