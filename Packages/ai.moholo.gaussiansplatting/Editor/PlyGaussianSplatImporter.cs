#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEngine;

#if UNITY_2020_2_OR_NEWER
using UnityEditor.AssetImporters;
#else
using UnityEditor.Experimental.AssetImporters;
#endif

namespace GaussianSplatting.Editor
{
    [ScriptedImporter(3, "ply")]
    public sealed class PlyGaussianSplatImporter : ScriptedImporter
    {
        [Tooltip("How to convert coordinates from the PLY file to Unity. Try None first.")]
        public CoordinateConversion coordConversion = CoordinateConversion.None;

        [Tooltip("Force refresh of the asset when settings change.")]
        public bool forceRefresh = false;

        public override void OnImportAsset(AssetImportContext ctx)
        {
            var asset = GaussianSplatAsset.CreateFromPlyFile(ctx.assetPath, coordConversion);
            asset.name = Path.GetFileNameWithoutExtension(ctx.assetPath);

            ctx.AddObjectToAsset("GaussianSplatAsset", asset);
            ctx.SetMainObject(asset);
            
            Debug.Log($"[PlyGaussianSplatImporter] Imported {asset.Count} splats from {ctx.assetPath} (conversion={coordConversion})");
        }
    }
}
#endif


