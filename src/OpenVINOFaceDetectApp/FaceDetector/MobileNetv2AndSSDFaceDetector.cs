using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Avalonia;
using OpenVINOFaceDetectApp.FaceDetector.Models;
using Sdcb.OpenVINO;
using SkiaSharp;

namespace OpenVINOFaceDetectApp.FaceDetector
{
    /// <summary>
    /// 特征网络：MobileNetv2，检测头：SSD
    /// </summary>
    public class MobileNetv2AndSSDFaceDetector : BaseFaceDetector
    {
        public MobileNetv2AndSSDFaceDetector(DeviceType deviceType) : base("./runtimes/models/face-detection-0204/FP32/face-detection-0204.xml", deviceType)
        {

        }

        public const string Name = "MobileNetv2AndSSD";

        public override List<DetectionResult> Detect(SKBitmap bitmap)
        {
            if (bitmap == null)
                return new List<DetectionResult>();
            if (InferRequest == null)
                return new List<DetectionResult>();

            using (Tensor? tensor = ToTensor(bitmap))
            {
                if (tensor == null) throw new Exception("Can not convert bitmap to tensor object.");
                InferRequest.Inputs.Primary = tensor;
            }

            InferRequest.Run();

            using Tensor output = InferRequest.Outputs.Primary;
            ReadOnlySpan<float> result = output.GetData<float>();
            int boxNum = output.Shape[3];
            List<DetectionResult> results = new();
            for (int i = 0; i < output.Shape[2]; ++i)
            {
                float outputConfidence = result[i * boxNum + 2];
                int clsId = (int)result[i * boxNum + 1];

                int x1 = (int)(result[i * boxNum + 3] * bitmap.Width);
                int y1 = (int)(result[i * boxNum + 4] * bitmap.Height);
                int x2 = (int)(result[i * boxNum + 5] * bitmap.Width);
                int y2 = (int)(result[i * boxNum + 6] * bitmap.Height);
                results.Add(new DetectionResult(clsId, outputConfidence, x1, y1, x2 - x1, y2 - y1));
            }
            return results;
        }

        public override void Dispose()
        {
            base.Dispose();
        }
    }
}
