using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SkiaSharp;

namespace OpenVINOFaceDetectApp.FaceDetector.Utils
{
    public static class SkiaSharpImageConvert
    {
        /// <summary>
        /// <see cref="Bitmap"/> 转为 3*8bit BGR <see cref="byte"/> 数组。
        /// </summary>
        /// <param name="source">待转换图像</param>
        /// <param name="width">图像宽度</param>
        /// <param name="height">图像高度</param>
        /// <param name="channels">图像通道</param>
        /// <returns>图像的 BGR <see cref="byte"/> 数组</returns>
        public static byte[] To24BGRByteArray(SKBitmap source, out int width, out int height, out int channels)
        {
            if (source == null)
            {
                throw new ArgumentNullException(nameof(source));
            }
            channels = 3;
            if (source.ColorType != SKColorType.Bgra8888)
            {
                using (SKBitmap bitmap = ConvertToBgra8888(source))
                {
                    width = bitmap.Width;
                    height = bitmap.Height;
                    return ConvertToByte(bitmap, channels);
                }
            }
            else
            {
                width = source.Width;
                height = source.Height;
                return ConvertToByte(source, channels);
            }
        }

        public static float[] To24BGRFloatArray(SKBitmap source, out int width, out int height, out int channels)
        {
            if (source == null)
            {
                throw new ArgumentNullException(nameof(source));
            }
            channels = 3;
            if (source.ColorType != SKColorType.Bgra8888)
            {
                using (SKBitmap bitmap = ConvertToBgra8888(source))
                {
                    width = bitmap.Width;
                    height = bitmap.Height;
                    return ConvertToFloatArray(bitmap);
                }
            }
            else
            {
                width = source.Width;
                height = source.Height;
                return ConvertToFloatArray(source);
            }
        }

        /// <summary>
        /// <see cref="Bitmap"/> 转为 3*8bit RGB <see cref="byte"/> 数组。
        /// </summary>
        /// <param name="source">待转换图像</param>
        /// <param name="width">图像宽度</param>
        /// <param name="height">图像高度</param>
        /// <param name="channels">图像通道</param>
        /// <returns>图像的 BGR <see cref="byte"/> 数组</returns>
        public static byte[] To24RGBByteArray(SKBitmap source, out int width, out int height, out int channels)
        {
            if (source == null)
            {
                throw new ArgumentNullException(nameof(source));
            }
            channels = 3;
            if (source.ColorType != SKColorType.Rgba8888)
            {
                using (SKBitmap bitmap = ConvertToRgba8888(source))
                {
                    width = bitmap.Width;
                    height = bitmap.Height;
                    return ConvertToByte(bitmap, channels);
                }
            }
            else
            {
                width = source.Width;
                height = source.Height;
                return ConvertToByte(source, channels);
            }
        }

        /// <summary>
        /// 转换图像格式
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        private static SKBitmap ConvertToBgra8888(SKBitmap source)
        {
            if (!source.CanCopyTo(SKColorType.Bgra8888))
            {
                throw new Exception("Can not copy image color type to Bgra8888");
            }
            SKBitmap bitmap = new SKBitmap();
            source.CopyTo(bitmap, SKColorType.Bgra8888);
            if (bitmap == null)
            {
                throw new Exception("Copy image to Bgra8888 failed");
            }
            return bitmap;
        }

        /// <summary>
        /// 转换图像格式
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        private static SKBitmap ConvertToRgba8888(SKBitmap source)
        {
            if (!source.CanCopyTo(SKColorType.Rgba8888))
            {
                throw new Exception("Can not copy image color type to Rgba8888");
            }
            SKBitmap bitmap = new SKBitmap();
            source.CopyTo(bitmap, SKColorType.Rgba8888);
            if (bitmap == null)
            {
                throw new Exception("Copy image to Rgba8888 failed");
            }
            return bitmap;
        }

        /// <summary>
        /// 转为BGR Bytes
        /// </summary>
        /// <param name="source"></param>
        /// <param name="channels"></param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        private static byte[] ConvertToByte(SKBitmap source, int channels)
        {
            byte[] array = source.Bytes;
            if (array == null || array.Length == 0)
            {
                throw new Exception("SKBitmap data is null");
            }
            byte[] bgra = new byte[array.Length / 4 * channels];
            // brga
            int j = 0;
            for (int i = 0; i < array.Length; i++)
            {
                if ((i + 1) % 4 == 0) continue;
                bgra[j] = array[i];
                j++;
            }
            return bgra;
        }

        private static float[] ConvertToFloatArray(SKBitmap source)
        {
            int pixelCount = source.Width * source.Height * 3;
            float[] pixelData = new float[pixelCount];
            int index = 0;
            for (int y = 0; y < source.Height; y++)
            {
                for (int x = 0; x < source.Width; x++)
                {
                    SKColor color = source.GetPixel(x, y);
                    float b = color.Blue / 255f;
                    float g = color.Green / 255f;
                    float r = color.Red / 255f;

                    // 存储 B、G、R 通道值到数组中
                    pixelData[index] = b;
                    pixelData[index + 1] = g;
                    pixelData[index + 2] = r;

                    index += 3;
                }
            }
            return pixelData;
        }
    }
}
