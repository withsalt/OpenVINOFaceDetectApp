namespace OpenVINOFaceDetectApp.ViewModels
{
    public class MainWindowViewModel : ViewModelBase
    {
        public CaptureControlViewModel CaptureControl { get; } = new CaptureControlViewModel();

        public void OnWindowOpened()
        {

        }
    }
}
