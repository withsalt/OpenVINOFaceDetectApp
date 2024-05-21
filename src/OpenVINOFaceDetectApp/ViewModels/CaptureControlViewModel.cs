using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Input;
using DynamicData;
using Epoxy;
using FlashCap;
using ReactiveUI;
using SkiaSharp;
using OpenVINOFaceDetectApp.Models;
using MsBox.Avalonia;
using MsBox.Avalonia.Enums;
using OpenVINOFaceDetectApp.FaceDetector;
using OpenVINOFaceDetectApp.FaceDetector.Models;

namespace OpenVINOFaceDetectApp.ViewModels
{
    public class CaptureControlViewModel : ReactiveObject
    {
        public CaptureControlViewModel()
        {
            this.WhenAnyValue(o => o.IsEnabled)
                .Subscribe(o =>
                {
                    this.RaisePropertyChanged(nameof(IsEnabled));
                });

            this.WhenAnyValue(o => o.Device)
                .Subscribe(o =>
                {
                    this.RaisePropertyChanged(nameof(Device));
                    this.RaisePropertyChanged(nameof(DeviceList));

                    InitDeviceList(o);
                });

            this.WhenAnyValue(o => o.Characteristics)
                .Subscribe(o =>
                {
                    this.RaisePropertyChanged(nameof(Characteristics));
                    this.RaisePropertyChanged(nameof(CharacteristicsList));
                });

            this.WhenAnyValue(o => o.BtnName)
                .Subscribe(o =>
                {
                    this.RaisePropertyChanged(nameof(BtnName));
                });

            StartBtnCommand = ReactiveCommand.Create(BtnAction);

            //加载默认字体
            using var stream = Avalonia.Platform.AssetLoader.Open(new Uri($"avares://{GetType().Assembly.GetName().Name}/Assets/Fonts/MSYH.TTC"));
            this.SKTypeface = SKTypeface.FromStream(stream);

            //加载识别器
            DetectorList.Add(Yolov8nFaceDetector.Name);
            DetectorList.Add(MobileNetv2AndSSDFaceDetector.Name);
            DetectorList.Add(ResNet152AndATSSFaceDetector.Name);

            this.Detector = DetectorList.First();
        }

        private IFaceDetector? _detectorHandle = null;

        #region Commonds

        public ICommand StartBtnCommand { get; private set; }

        #endregion

        #region 属性

        private SKTypeface SKTypeface { get; }

        public DetectOptions DetectOptions { get; private set; } = new DetectOptions();

        public ObservableCollection<CaptureDeviceDescriptor?> DeviceList { get; } = new();

        public CaptureDeviceDescriptor? _device = null;
        public CaptureDeviceDescriptor? Device
        {
            get => this._device;
            set
            {
                this.RaiseAndSetIfChanged(ref _device, value);
            }
        }

        public ObservableCollection<string?> DetectorList { get; } = new();


        public string? _detector = null;
        public string? Detector
        {
            get => this._detector;
            set
            {
                this.RaiseAndSetIfChanged(ref _detector, value);
            }
        }

        public ObservableCollection<VideoCharacteristics> CharacteristicsList { get; } = new();

        private VideoCharacteristics? _characteristics = null;
        public VideoCharacteristics? Characteristics
        {
            get => this._characteristics;
            set => this.RaiseAndSetIfChanged(ref _characteristics, value);
        }

        private bool _isEnabled = false;

        public bool IsEnabled
        {
            get => _isEnabled;
            set => this.RaiseAndSetIfChanged(ref _isEnabled, value);
        }

        private bool _isEnabledDeviceSelector = false;

        public bool IsEnabledDeviceSelector
        {
            get => this.IsEnabled && _isEnabledDeviceSelector;
            set => this.RaiseAndSetIfChanged(ref _isEnabledDeviceSelector, value);
        }

        public string? _btnName = "Not Ready";
        public string? BtnName
        {
            get => _btnName;
            set => this.RaiseAndSetIfChanged(ref _btnName, value);
        }

        public string? _btnBackground = "Gainsboro";
        public string? BtnBackground
        {
            get => _btnBackground;
            set => this.RaiseAndSetIfChanged(ref _btnBackground, value);
        }

        public SKBitmap? _image = null;
        public SKBitmap? Image
        {
            get => _image;
            set => this.RaiseAndSetIfChanged(ref _image, value);
        }

        #endregion

        private long _countFrames;
        private enum States
        {
            Ready,
            Running,
            Stopped
        }

        private States _state = States.Stopped;

        private void UpdateCurrentState(States state)
        {
            this._state = state;

            switch (this._state)
            {
                case States.Ready:
                    {
                        this.IsEnabled = true;
                        this.IsEnabledDeviceSelector = true;
                        this.BtnName = "Start";
                        this.BtnBackground = "GreenYellow";
                    }
                    break;
                case States.Running:
                    {
                        this.BtnName = "Stop";
                        this.BtnBackground = "Red";
                        this.IsEnabledDeviceSelector = false;
                    }
                    break;
                case States.Stopped:
                    {
                        this.BtnName = "Start";
                        this.BtnBackground = "GreenYellow";
                        this.IsEnabledDeviceSelector = true;
                        if (this.Image != null)
                        {
                            try { this.Image.Dispose(); } catch { }
                            this.Image = new SKBitmap(100, 100);
                            this.Image.Dispose();
                        }
                    }
                    break;
                default:

                    break;
            }
        }

        private CaptureDevice? _captureDevice = null;

        private async Task BtnAction()
        {
            if (_state == States.Stopped || _state == States.Ready)
            {
                await StartAsync();
            }
            else
            {
                await StopAsync();
            }
        }

        private async Task StartAsync()
        {
            if (this.Device == null)
            {
                await MessageBoxManager.GetMessageBoxStandard("错误", "请指定视频输入设备", ButtonEnum.Ok, Icon.Warning).ShowAsync();
                return;
            }
            if (this.Characteristics == null)
            {
                await MessageBoxManager.GetMessageBoxStandard("提示", "请指定视频输入分辨率", ButtonEnum.Ok, Icon.Warning).ShowAsync();
                return;
            }
            try
            {
                if (_detectorHandle == null)
                {
                    _detectorHandle = GetFaceDetector();
                }
                _captureDevice = await this.Device.OpenAsync(this.Characteristics, this.OnPixelBufferArrivedAsync);
                if (_captureDevice == null)
                {
                    await MessageBoxManager.GetMessageBoxStandard("提示", $"无法打开指定的设备：{this.Device.Name}", ButtonEnum.Ok, Icon.Warning).ShowAsync();
                    return;
                }
                await _captureDevice.StartAsync();
                UpdateCurrentState(States.Running);
            }
            catch (Exception ex)
            {
                _captureDevice = null;
                await MessageBoxManager.GetMessageBoxStandard("提示", $"无法打开指定的设备：{this.Device.Name}，{ex.Message}", ButtonEnum.Ok, Icon.Warning).ShowAsync();
                return;
            }
        }

        private IFaceDetector GetFaceDetector()
        {
            switch (_detector)
            {
                case ResNet152AndATSSFaceDetector.Name:
                    return new ResNet152AndATSSFaceDetector(DeviceType.CPU);
                case MobileNetv2AndSSDFaceDetector.Name:
                    return new MobileNetv2AndSSDFaceDetector(DeviceType.CPU);
                case Yolov8nFaceDetector.Name:
                    return new Yolov8nFaceDetector(DeviceType.CPU);
                default:
                    throw new NotImplementedException("未知的检测器类型");
            }
        }

        private async Task StopAsync()
        {
            try
            {
                if (_captureDevice == null) return;
                await _captureDevice.StopAsync();
                UpdateCurrentState(States.Stopped);
            }
            finally
            {
                _captureDevice?.Dispose();
                _captureDevice = null;
                if (_detectorHandle != null)
                {
                    _detectorHandle.Dispose();
                    _detectorHandle = null;
                }
            }
        }

        public void InitDeviceList(CaptureDeviceDescriptor? captureDevice = null)
        {
            if (captureDevice == null)
            {
                var devices = new CaptureDevices();
                this.DeviceList.Clear();
                var canListDevice = devices.EnumerateDescriptors().Where(d => d.Characteristics.Length >= 1);
                if (canListDevice?.Any() != true)
                {
                    return;
                }
                this.DeviceList.AddRange(canListDevice);
                this.Device = this.DeviceList.FirstOrDefault();

                VideoCharacteristics[]? characteristics = this.Device?.Characteristics;
                if (characteristics?.Any() != true)
                {
                    return;
                }
                CharacteristicsList.AddRange(characteristics);
                UpdateCurrentState(States.Ready);
                var targetCharacteristics = CharacteristicsList.FirstOrDefault(p => p.Width == 1280 && p.Height == 720 && p.PixelFormat == PixelFormats.JPEG);
                if (targetCharacteristics != null)
                {
                    this.Characteristics = targetCharacteristics;
                }
            }
            else
            {
                VideoCharacteristics[]? characteristics = this.Device?.Characteristics;
                if (characteristics?.Any() != true)
                {
                    this.CharacteristicsList.Clear();
                    return;
                }
                VideoCharacteristics? targetCharacteristics = null;
                if (this.Characteristics != null)
                {
                    targetCharacteristics = characteristics.FirstOrDefault(p => p.Width == this.Characteristics.Width && p.Height == this.Characteristics.Height && p.PixelFormat == this.Characteristics.PixelFormat);
                    if (targetCharacteristics == null)
                    {
                        targetCharacteristics = characteristics.FirstOrDefault(p => p.PixelFormat != PixelFormats.Unknown);
                    }
                }
                else
                {
                    targetCharacteristics = characteristics.FirstOrDefault(p => p.Width == 1280 && p.Height == 720 && p.PixelFormat == PixelFormats.JPEG);
                }
                this.CharacteristicsList.Clear();
                this.CharacteristicsList.AddRange(characteristics);
                if (targetCharacteristics != null)
                {
                    this.Characteristics = targetCharacteristics;
                }
            }
        }

        private async Task OnPixelBufferArrivedAsync(PixelBufferScope bufferScope)
        {
            try
            {
                ArraySegment<byte> image = bufferScope.Buffer.ReferImage();
                // Decode image data
                var bitmap = SKBitmap.Decode(image);
                // Capture statistics variables.
                var countFrames = Interlocked.Increment(ref this._countFrames);
                var frameIndex = bufferScope.Buffer.FrameIndex;
                var timestamp = bufferScope.Buffer.Timestamp;

                if (_state == States.Running && await UIThread.TryBind())
                {
                    this.Image?.Dispose();
                    var realFps = countFrames / timestamp.TotalSeconds;
                    FaceDetect(bitmap, realFps);
                    this.Image = bitmap;
                }
            }
            finally
            {
                bufferScope.ReleaseNow();
            }
        }

        private int ObtainDynamic(int baseNum)
        {
            if (this.Characteristics == null)
            {
                return 1;
            }
            int result = (int)(this.Characteristics.Height / 720.0 * baseNum);
            if (result <= 0)
            {
                result = 1;
            }
            return result;
        }

        private void FaceDetect(SKBitmap bitmap, double fps)
        {
            if (_detectorHandle == null)
            {
                return;
            }

            List<DetectionResult> results = _detectorHandle.Detect(bitmap);
            if (results.Count == 0)
            {
                return;
            }

            foreach (var item in results)
            {
                if (item.Confidence < 0.6f)
                {
                    continue;
                }
                using (SKCanvas canvas = new SKCanvas(bitmap))
                {
                    using (SKPaint paint = new SKPaint())
                    {
                        paint.Style = SKPaintStyle.Stroke;
                        paint.Color = SKColors.Red;
                        paint.StrokeWidth = ObtainDynamic(3);
                        paint.StrokeCap = SKStrokeCap.Round;
                        canvas.DrawRect(item.X, item.Y, item.Width, item.Height, paint);

                        if (item.Points?.Any() == true)
                        {
                            paint.Style = SKPaintStyle.Fill;
                            foreach (var point in item.Points)
                            {
                                int obWidth = ObtainDynamic(10);
                                canvas.DrawRect(point.X, point.Y, obWidth, obWidth, paint);
                            }
                        }
                    }
                }
            }
        }
    }
}
