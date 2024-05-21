using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OpenVINOFaceDetectApp.FaceDetector.Models
{
    public record DetectionResult
    {
        public int ClassId { get; set; } = 0;

        public float Confidence { get; set; } = 0.0f;

        public float X { get; set; } = 0.0f;

        public float Y { get; set; } = 0.0f;

        public float Width { get; set; } = 0.0f;

        public float Height { get; set; } = 0.0f;

        public IEnumerable<Point>? Points { get; set; }

        public DetectionResult(int classId, float confidence, float x, float y, float width, float height, List<Point>? points = null)
        {
            ClassId = classId;
            Confidence = confidence;
            X = x;
            Y = y;
            Width = width;
            Height = height;
            if (points?.Any() == true)
            {
                this.Points = points;
            }
        }
    }
}
