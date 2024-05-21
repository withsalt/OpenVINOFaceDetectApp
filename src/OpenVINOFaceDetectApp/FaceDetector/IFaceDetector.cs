using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenVINOFaceDetectApp.FaceDetector.Models;
using SkiaSharp;

namespace OpenVINOFaceDetectApp.FaceDetector
{
    public interface IFaceDetector : IDisposable
    {
        List<DetectionResult> Detect(SKBitmap bitmap);
    }
}
