using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OpenVINOFaceDetectApp.Models
{
    public class DetectOptions
    {
        public bool IsTrack { get; set; } = true;

        public bool PropertyDetect { get; set; } = true;
    }
}
