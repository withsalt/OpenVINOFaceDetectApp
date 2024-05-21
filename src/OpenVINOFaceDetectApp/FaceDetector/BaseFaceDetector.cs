using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;
using OpenVINOFaceDetectApp.FaceDetector.Models;
using OpenVINOFaceDetectApp.FaceDetector.Utils;
using Sdcb.OpenVINO;
using Sdcb.OpenVINO.Natives;
using SkiaSharp;

namespace OpenVINOFaceDetectApp.FaceDetector
{
    public abstract class BaseFaceDetector : IFaceDetector
    {
        protected InferRequest InferRequest { get; private set; }

        protected Size InputSize { get; private set; }

        protected string ModelPath { get; private set; }

        public BaseFaceDetector(string modelPath, DeviceType deviceType)
        {
            if (string.IsNullOrEmpty(modelPath) || !File.Exists(modelPath))
            {
                throw new FileNotFoundException($"The model file '{modelPath}' not exist.");
            }
            if (!Enum.IsDefined(typeof(DeviceType), deviceType))
            {
                throw new ArgumentOutOfRangeException($"Can not support device type {deviceType}");
            }
            var initResult = Init(Path.GetFullPath(modelPath), deviceType);
            this.InferRequest = initResult.Item1;
            this.InputSize = initResult.Item2;
            this.ModelPath = modelPath;
        }

        public abstract List<DetectionResult> Detect(SKBitmap bitmap);

        protected virtual Tensor? ToTensor(SKBitmap bitmap)
        {
            if (bitmap == null) return null;
            using var targetBitmap = bitmap.Resize(new SKSizeI(this.InputSize.Width, this.InputSize.Height), SKFilterQuality.High);
            if (targetBitmap == null) return null;
            var bytes = SkiaSharpImageConvert.To24BGRByteArray(targetBitmap, out _, out _, out _);
            var tensor = Tensor.FromArray(bytes, this.InferRequest.Inputs.Primary.Shape);
            return tensor;
        }

        protected virtual (InferRequest, Size) Init(string modelPath, DeviceType deviceType)
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
            Layout inputLayout = Layout.NHWC;
            using Model rawModel = OVCore.Shared.ReadModel(modelPath);
            using PrePostProcessor pp = rawModel.CreatePrePostProcessor();
            using (PreProcessInputInfo inputInfo = pp.Inputs.Primary)
            {
                inputInfo.TensorInfo.Layout = inputLayout;
                inputInfo.TensorInfo.ElementType = ov_element_type_e.U8;
                inputInfo.ModelInfo.Layout = Layout.NCHW;
            }
            using Model model = pp.BuildModel();
            using CompiledModel cm = OVCore.Shared.CompileModel(model, device);

            InferRequest inferRequest = cm.CreateInferRequest();
            if (inferRequest == null)
            {
                throw new Exception("Create infer request failed.");
            }
            Size size = new Size();
            if (inputLayout.ToString() == Layout.NHWC.ToString())
            {
                size = new Size(inferRequest.Inputs.Primary.Shape[2], inferRequest.Inputs.Primary.Shape[1]);
            }
            else if (inputLayout.ToString() == Layout.NCHW.ToString())
            {
                size = new Size(inferRequest.Inputs.Primary.Shape[3], inferRequest.Inputs.Primary.Shape[2]);
            }
            return (inferRequest, size);
        }

        public virtual void Dispose()
        {
            if (this.InferRequest != null && !this.InferRequest.Disposed)
            {
                this.InferRequest.Dispose();
            }
        }

        protected string GetDeviceName(DeviceType value)
        {
            FieldInfo? field = value.GetType().GetField(value.ToString());
            if (field != null)
            {
                var attribute = Attribute.GetCustomAttribute(field, typeof(DescriptionAttribute));
                if (attribute != null)
                {
                    return ((DescriptionAttribute)attribute).Description;
                }
            }
            return value.ToString();
        }
    }
}
