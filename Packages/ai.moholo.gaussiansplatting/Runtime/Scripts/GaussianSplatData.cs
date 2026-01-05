using System;
using UnityEngine;

namespace GaussianSplatting
{
    [System.Serializable]
    public class GaussianSplatData
    {
        public int Count;
        public Vector3[] Centers = Array.Empty<Vector3>();
        public Vector4[] Rotations = Array.Empty<Vector4>(); // (x,y,z,w)
        public Vector3[] Scales = Array.Empty<Vector3>();    // exp() already applied
        public Vector4[] Colors = Array.Empty<Vector4>();    // rgb in [0..1], a in [0..1]

        public int ShBands; // 0..3
        public int ShCoeffsPerSplat; // 0, 3, 8, 15
        public Vector3[] ShCoeffs = Array.Empty<Vector3>(); // length = Count * ShCoeffsPerSplat

        public GaussianSplatData() { }

        public GaussianSplatData(GaussianSplatData other)
        {
            if (other == null) return;
            Count = other.Count;
            Centers = other.Centers != null ? (Vector3[])other.Centers.Clone() : Array.Empty<Vector3>();
            Rotations = other.Rotations != null ? (Vector4[])other.Rotations.Clone() : Array.Empty<Vector4>();
            Scales = other.Scales != null ? (Vector3[])other.Scales.Clone() : Array.Empty<Vector3>();
            Colors = other.Colors != null ? (Vector4[])other.Colors.Clone() : Array.Empty<Vector4>();
            ShBands = other.ShBands;
            ShCoeffsPerSplat = other.ShCoeffsPerSplat;
            ShCoeffs = other.ShCoeffs != null ? (Vector3[])other.ShCoeffs.Clone() : Array.Empty<Vector3>();
        }
    }
}