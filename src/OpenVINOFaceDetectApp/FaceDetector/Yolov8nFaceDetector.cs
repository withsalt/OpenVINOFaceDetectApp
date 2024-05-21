using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using OpenVINOFaceDetectApp.FaceDetector.Models;
using OpenVINOFaceDetectApp.FaceDetector.Utils;
using Sdcb.OpenVINO;
using Sdcb.OpenVINO.Natives;
using SkiaSharp;


namespace OpenVINOFaceDetectApp.FaceDetector
{
    /// <summary>
    /// yolov8-face https://github.com/derronqi/yolov8-face
    /// </summary>
    public class Yolov8nFaceDetector : BaseFaceDetector
    {
        const int num_class = 1;
        const int reg_max = 16;
        public Yolov8nFaceDetector(DeviceType deviceType) : base("./runtimes/models/yolov8n-face/yolov8n-face.onnx", deviceType)
        {

        }

        public const string Name = "Yolov8nFace";

        protected override (InferRequest, Size) Init(string modelPath, DeviceType deviceType)
        {
            var devices = OVCore.Shared.AvailableDevices;
            if (devices?.Any() != true)
            {
                throw new Exception("No available devices");
            }
            string device = GetDeviceName(deviceType);
            if (!devices.Contains(device))
            {
                throw new Exception($"{device} is not available.");
            }
            using Model rawModel = OVCore.Shared.ReadModel(modelPath);
            using PrePostProcessor pp = rawModel.CreatePrePostProcessor();
            using (PreProcessInputInfo inputInfo = pp.Inputs.Primary)
            {
                inputInfo.TensorInfo.Layout = Layout.NHWC;
                inputInfo.TensorInfo.ElementType = ov_element_type_e.F32;
                inputInfo.ModelInfo.Layout = Layout.NCHW;
            }
            using Model model = pp.BuildModel();
            using CompiledModel cm = OVCore.Shared.CompileModel(model, device);

            InferRequest inferRequest = cm.CreateInferRequest();
            if (inferRequest == null)
            {
                throw new Exception("Create infer request failed.");
            }
            var size = new Size(inferRequest.Inputs.Primary.Shape[1], inferRequest.Inputs.Primary.Shape[1]);
            return (inferRequest, size);
        }

        protected override Tensor? ToTensor(SKBitmap bitmap)
        {
            if (bitmap == null) return null;
            using var targetBitmap = bitmap.Resize(new SKSizeI(this.InputSize.Width, this.InputSize.Height), SKFilterQuality.High);
            if (targetBitmap == null) return null;
            var bytes = SkiaSharpImageConvert.To24BGRFloatArray(targetBitmap, out _, out _, out _);
            var tensor = Tensor.FromArray(bytes, this.InferRequest.Inputs.Primary.Shape);
            return tensor;
        }

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

            var ry = bitmap.Height / (double)this.InputSize.Height;
            var rx = bitmap.Width / (double)this.InputSize.Width;

            List<DetectionResult> results = new();

            SizeF sizeRatio = new SizeF(1f * bitmap.Width / this.InputSize.Width, 1f * bitmap.Height / this.InputSize.Height);
            List<Rectangle> boxes = new List<Rectangle>();
            List<List<Point>> landmarks = new List<List<Point>>();
            List<float> confidences = new List<float>();

            foreach (var output in this.InferRequest.Outputs)
            {
                try
                {
                    ReadOnlySpan<float> data = output.GetData<float>();
                    GenerateProposal(data, output.Shape, boxes, confidences, landmarks, bitmap.Height, bitmap.Width, sizeRatio.Height, sizeRatio.Width);
                }
                finally
                {
                    output.Dispose();
                }
            }
            NMSBoxes(boxes, confidences, 0.5f, 0.5f, out int[] indices);
            if (indices.Length > 0)
            {
                for (int i = 0; i < indices.Length; i++)
                {
                    results.Add(new DetectionResult(0, confidences[indices[i]], boxes[indices[i]].X, boxes[indices[i]].Y, boxes[indices[i]].Width, boxes[indices[i]].Height, landmarks[indices[i]]));
                }
            }
            return results;
        }

        private void GenerateProposal(ReadOnlySpan<float> tensorData, Shape shape, List<Rectangle> boxes, List<float> confidences, List<List<Point>> landmarks, int imgh, int imgw, float ratioh, float ratiow)
        {
            int feat_h = shape[2];
            int feat_w = shape[3];
            int stride = (int)Math.Ceiling((float)this.InputSize.Height / feat_h);
            int area = feat_h * feat_w;
            int clsIndex = area * reg_max * 4;
            int kpIndex = area * (reg_max * 4 + 1);
            var ptr_cls = tensorData.Slice(clsIndex, tensorData.Length - clsIndex);
            var ptr_kp = tensorData.Slice(kpIndex, tensorData.Length - kpIndex);

            for (int i = 0; i < feat_h; i++)
            {
                for (int j = 0; j < feat_w; j++)
                {
                    int index = i * feat_w + j;
                    int cls_id = -1;
                    float max_conf = -10000;
                    for (int k = 0; k < num_class; k++)
                    {
                        float conf = ptr_cls[k * area + index];
                        if (conf > max_conf)
                        {
                            max_conf = conf;
                            cls_id = k;
                        }
                    }
                    float box_prob = SigmoidX(max_conf);
                    if (box_prob > 0.5)
                    {
                        float[] pred_ltrb = new float[4];
                        float[] dfl_value = new float[reg_max];
                        float[] dfl_softmax = new float[reg_max];
                        for (int k = 0; k < 4; k++)
                        {
                            for (int n = 0; n < reg_max; n++)
                            {
                                dfl_value[n] = tensorData[(k * reg_max + n) * area + index];
                            }
                            Softmax(dfl_value, dfl_softmax, reg_max);

                            float dis = 0f;
                            for (int n = 0; n < reg_max; n++)
                            {
                                dis += n * dfl_softmax[n];
                            }

                            pred_ltrb[k] = dis * stride;
                        }
                        float cx = (j + 0.5f) * stride;
                        float cy = (i + 0.5f) * stride;
                        float xmin = Math.Max((cx - pred_ltrb[0]) * ratiow, 0f);
                        float ymin = Math.Max((cy - pred_ltrb[1]) * ratioh, 0f);
                        float xmax = Math.Min((cx + pred_ltrb[2]) * ratiow, imgw - 1);
                        float ymax = Math.Min((cy + pred_ltrb[3]) * ratioh, imgh - 1);

                        Rectangle box = new Rectangle((int)xmin, (int)ymin, (int)(xmax - xmin), (int)(ymax - ymin));
                        boxes.Add(box);
                        confidences.Add(box_prob);

                        List<Point> kpts = new List<Point>(5);
                        for (int k = 0; k < 5; k++)
                        {
                            float x = (((ptr_kp[(k * 3) * area + index] * 2 + j) * stride) * ratiow);
                            float y = (((ptr_kp[(k * 3 + 1) * area + index] * 2 + i) * stride) * ratioh);
                            // float pt_conf = SigmoidX(ptr_kp[(k * 3 + 2) * area + index]);
                            kpts.Add(new Point((int)x, (int)y));
                        }
                        landmarks.Add(kpts);
                    }
                }
            }
        }

        private float SigmoidX(float x)
        {
            return (float)(1.0 / (1.0 + Math.Exp(-x)));
        }

        private void Softmax(float[] x, float[] y, int length)
        {
            float sum = 0;
            for (int i = 0; i < length; i++)
            {
                y[i] = (float)Math.Exp(x[i]);
                sum += y[i];
            }
            for (int i = 0; i < length; i++)
            {
                y[i] /= sum;
            }
        }

        /// <summary>
        /// 用于非最大抑制，以便在目标检测或物体识别中消除重叠的边界框。
        /// (From chatgpt, chatgpt yes!)
        /// </summary>
        /// <param name="bboxes"></param>
        /// <param name="scores"></param>
        /// <param name="scoreThreshold"></param>
        /// <param name="nmsThreshold"></param>
        /// <param name="indices"></param>
        /// <param name="eta"></param>
        /// <param name="topK"></param>
        /// <returns></returns>
        private Rectangle[] NMSBoxes(IEnumerable<Rectangle> bboxes, IEnumerable<float> scores, float scoreThreshold, float nmsThreshold, out int[] indices, float eta = 1f, int topK = 0)
        {
            // 创建一个空列表来存储最终的边界框
            List<Rectangle> selectedRects = new List<Rectangle>();

            // 将bboxes和scores转换为数组以便索引
            Rectangle[] bboxesArray = bboxes.ToArray();
            float[] scoresArray = scores.ToArray();

            // 根据分数阈值过滤边界框，并记录它们的索引
            List<int> validIndices = new List<int>();
            for (int i = 0; i < scoresArray.Length; i++)
            {
                if (scoresArray[i] > scoreThreshold)
                {
                    validIndices.Add(i);
                }
            }

            // 根据分数降序排序索引
            validIndices.Sort((a, b) => scoresArray[b].CompareTo(scoresArray[a]));

            // 非最大抑制
            List<int> selectedIndices = new List<int>();
            while (validIndices.Count > 0)
            {
                int currentIdx = validIndices[0];
                selectedIndices.Add(currentIdx);
                validIndices.RemoveAt(0);

                for (int i = 0; i < validIndices.Count;)
                {
                    float overlap = CalculateOverlap(bboxesArray[currentIdx], bboxesArray[validIndices[i]]);
                    if (overlap > nmsThreshold)
                    {
                        validIndices.RemoveAt(i);
                    }
                    else
                    {
                        i++;
                    }
                }
            }

            // 如果topK大于0，则保留前topK个索引
            if (topK > 0 && selectedIndices.Count > topK)
            {
                selectedIndices.RemoveRange(topK, selectedIndices.Count - topK);
            }

            // 将结果存储到输出参数indices中
            indices = selectedIndices.ToArray();

            // 根据选定的索引构建最终的边界框列表
            foreach (int idx in selectedIndices)
            {
                selectedRects.Add(bboxesArray[idx]);
            }

            return selectedRects.ToArray();
        }

        // 计算两个矩形的重叠率
        private float CalculateOverlap(Rectangle rect1, Rectangle rect2)
        {
            int x1 = Math.Max(rect1.Left, rect2.Left);
            int y1 = Math.Max(rect1.Top, rect2.Top);
            int x2 = Math.Min(rect1.Right, rect2.Right);
            int y2 = Math.Min(rect1.Bottom, rect2.Bottom);

            int intersectionArea = Math.Max(0, x2 - x1 + 1) * Math.Max(0, y2 - y1 + 1);
            int rect1Area = rect1.Width * rect1.Height;
            int rect2Area = rect2.Width * rect2.Height;

            float overlap = (float)intersectionArea / (rect1Area + rect2Area - intersectionArea);
            return overlap;
        }

        public override void Dispose()
        {
            base.Dispose();
        }
    }
}
