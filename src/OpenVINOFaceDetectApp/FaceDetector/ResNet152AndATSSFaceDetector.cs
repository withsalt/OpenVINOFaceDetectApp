using System;
using System.Collections.Generic;
using System.Drawing;
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
    /// 特征网络：ResNet152，检测头：ATSS
    /// </summary>
    public class ResNet152AndATSSFaceDetector : BaseFaceDetector
    {
        //需要自行下载
        public ResNet152AndATSSFaceDetector(DeviceType deviceType) : base("./runtimes/models/face-detection-0206/FP32/face-detection-0206.xml", deviceType)
        {

        }

        public const string Name = "ResNet152AndATSS";

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
            SizeF sizeRatio = new SizeF(1f * bitmap.Width / this.InputSize.Width, 1f * bitmap.Height / this.InputSize.Height);
            List<DetectionResult> results = new();
            foreach (var outputsItem in InferRequest.Outputs)
            {
                try
                {
                    if (outputsItem.Size < 5)
                        continue;
                    if (outputsItem.Shape.Rank < 2)
                        continue;
                    ReadOnlySpan<float> result = outputsItem.GetData<float>();
                    for (int i = 0; i < outputsItem.Shape[0]; ++i)
                    {
                        int boxNum = outputsItem.Shape[1];
                        float outputConfidence = result[i * boxNum + 4];
                        int clsId = 0;

                        int x1 = (int)(result[i * boxNum + 0] * sizeRatio.Width);
                        int y1 = (int)(result[i * boxNum + 1] * sizeRatio.Height);
                        int x2 = (int)(result[i * boxNum + 2] * sizeRatio.Width);
                        int y2 = (int)(result[i * boxNum + 3] * sizeRatio.Height);

                        results.Add(new(clsId, outputConfidence, x1, y1, x2 - x1, y2 - y1));
                    }
                }
                finally
                {
                    outputsItem.Dispose();
                }
            }
            return results;
        }

        public override void Dispose()
        {
            base.Dispose();
        }
    }
}
