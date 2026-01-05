using UnityEngine;

namespace GaussianSplatting
{
    public sealed class GaussianSplatAsset : ScriptableObject
    {
        [SerializeField] private GaussianSplatData _data = new GaussianSplatData();

        [Header("Metadata")]
        [Tooltip("Original source file path or URL")]
        public string SourcePath;
        [Tooltip("Pre-rendered thumbnail for editor preview")]
        public Texture2D Thumbnail;

        public GaussianSplatData Data
        {
            get => _data;
            set => _data = value ?? new GaussianSplatData();
        }

        public int Count => _data?.Count ?? 0;
        public Vector3[] Centers => _data?.Centers;
        public Vector4[] Rotations => _data?.Rotations;
        public Vector3[] Scales => _data?.Scales;
        public Vector4[] Colors => _data?.Colors;
        public int ShBands => _data?.ShBands ?? 0;
        public int ShCoeffsPerSplat => _data?.ShCoeffsPerSplat ?? 0;
        public Vector3[] ShCoeffs => _data?.ShCoeffs;

        public void SetData(GaussianSplatData data)
        {
            _data = data ?? new GaussianSplatData();
        }

        public GaussianSplatData GetDataCopy()
        {
            return new GaussianSplatData(_data);
        }

        public void LoadFromPly(byte[] plyBytes, CoordinateConversion conversion = CoordinateConversion.RightHandedToUnity)
        {
            var data = PlyGaussianSplatLoader.Load(plyBytes, conversion);
            SetData(data);
        }

        public void LoadFromPlyFile(string filePath, CoordinateConversion conversion = CoordinateConversion.RightHandedToUnity)
        {
            var bytes = System.IO.File.ReadAllBytes(filePath);
            LoadFromPly(bytes, conversion);
            SourcePath = filePath;
        }

        public static GaussianSplatAsset CreateFromPly(byte[] plyBytes, CoordinateConversion conversion = CoordinateConversion.RightHandedToUnity, string sourcePath = null)
        {
            var asset = CreateInstance<GaussianSplatAsset>();
            asset.LoadFromPly(plyBytes, conversion);
            asset.SourcePath = sourcePath;
            return asset;
        }

        public static GaussianSplatAsset CreateFromPlyFile(string filePath, CoordinateConversion conversion = CoordinateConversion.RightHandedToUnity)
        {
            var asset = CreateInstance<GaussianSplatAsset>();
            asset.LoadFromPlyFile(filePath, conversion);
            return asset;
        }
    }
}


