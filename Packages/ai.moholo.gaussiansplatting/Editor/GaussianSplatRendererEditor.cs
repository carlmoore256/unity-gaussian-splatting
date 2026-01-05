#if UNITY_EDITOR
using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace GaussianSplatting.Editor
{
    [InitializeOnLoad]
    public static class GaussianSplatSceneLoader
    {
        private static double _lastLoadTime = 0;
        private const double LOAD_DEBOUNCE_SECONDS = 0.5;

        static GaussianSplatSceneLoader()
        {
            EditorSceneManager.sceneOpened += OnSceneOpened;
            EditorApplication.hierarchyChanged += OnHierarchyChanged;
            EditorApplication.delayCall += () =>
            {
                if (!Application.isPlaying)
                    LoadAllRenderersInScene();
            };
        }

        private static void OnSceneOpened(UnityEngine.SceneManagement.Scene scene, OpenSceneMode mode)
        {
            EditorApplication.delayCall += LoadAllRenderersInScene;
        }

        private static void OnHierarchyChanged()
        {
            double currentTime = EditorApplication.timeSinceStartup;
            if (currentTime - _lastLoadTime < LOAD_DEBOUNCE_SECONDS) return;
            _lastLoadTime = currentTime;
            EditorApplication.delayCall += LoadAllRenderersInScene;
        }

        private static void LoadAllRenderersInScene()
        {
            if (Application.isPlaying) return;

            var renderers = UnityEngine.Object.FindObjectsByType<GaussianSplatRenderer>(FindObjectsInactive.Include, FindObjectsSortMode.None);
            if (renderers == null || renderers.Length == 0) return;

            foreach (var renderer in renderers)
            {
                if (renderer == null) continue;
                if (renderer.Asset != null && !renderer.IsLoaded && renderer.isActiveAndEnabled)
                {
                    renderer.LoadAssetToGPU();
                    EditorUtility.SetDirty(renderer);
                }
            }
            SceneView.RepaintAll();
        }
    }

    [CustomEditor(typeof(GaussianSplatRenderer))]
    public class GaussianSplatRendererEditor : UnityEditor.Editor
    {
        private string[] _availablePlyFiles = Array.Empty<string>();
        private int _selectedPlyIndex = 0;

        private void OnEnable()
        {
            RefreshPlyFileList();
        }

        private void RefreshPlyFileList()
        {
            string streamingPath = Path.Combine(Application.dataPath, "StreamingAssets", "GaussianSplatting");
            if (Directory.Exists(streamingPath))
            {
                _availablePlyFiles = Directory.GetFiles(streamingPath, "*.ply")
                    .Select(Path.GetFileName)
                    .OrderBy(f => f)
                    .ToArray();
            }
            else
            {
                _availablePlyFiles = Array.Empty<string>();
            }
        }

        public override void OnInspectorGUI()
        {
            if (!(target is GaussianSplatRenderer renderer)) return;

            serializedObject.Update();

            EditorGUILayout.LabelField("Gaussian Splat Asset", EditorStyles.boldLabel);
            
            EditorGUI.BeginChangeCheck();
            var newAsset = (GaussianSplatAsset)EditorGUILayout.ObjectField(
                "Asset", renderer.Asset, typeof(GaussianSplatAsset), false);
            if (EditorGUI.EndChangeCheck() && newAsset != renderer.Asset)
            {
                Undo.RecordObject(renderer, "Change Gaussian Splat Asset");
                renderer.Asset = newAsset;
                EditorUtility.SetDirty(renderer);
            }

            if (renderer.Asset != null)
            {
                EditorGUI.indentLevel++;
                EditorGUILayout.LabelField("Count", renderer.Asset.Count.ToString());
                if (!string.IsNullOrEmpty(renderer.Asset.SourcePath))
                    EditorGUILayout.LabelField("Source", renderer.Asset.SourcePath);
                EditorGUI.indentLevel--;

                EditorGUILayout.BeginHorizontal();
                if (GUILayout.Button(renderer.IsLoaded ? "Reload Asset" : "Load Asset"))
                {
                    renderer.LoadAssetToGPU();
                    SceneView.RepaintAll();
                }
                EditorGUILayout.EndHorizontal();
            }

            if (renderer.IsLoaded)
            {
                EditorGUILayout.HelpBox($"Loaded: {renderer.SplatCount} splats", MessageType.Info);
            }
            else
            {
                EditorGUILayout.HelpBox("Not loaded. Assign an asset or use RuntimePlyLoader.", MessageType.Warning);
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Quick Import from StreamingAssets", EditorStyles.boldLabel);
            
            if (_availablePlyFiles.Length == 0)
            {
                EditorGUILayout.HelpBox("No .ply files in Assets/StreamingAssets/GaussianSplatting/", MessageType.Info);
            }
            else
            {
                _selectedPlyIndex = Mathf.Clamp(_selectedPlyIndex, 0, _availablePlyFiles.Length - 1);
                _selectedPlyIndex = EditorGUILayout.Popup("PLY File", _selectedPlyIndex, _availablePlyFiles);
                
                if (GUILayout.Button("Import and Create Asset"))
                {
                    ImportPlyToAsset(renderer, _availablePlyFiles[_selectedPlyIndex]);
                }
            }

            if (GUILayout.Button("Refresh PLY List"))
                RefreshPlyFileList();

            EditorGUILayout.Space();
            DrawPropertiesExcluding(serializedObject, "m_Script", "_asset");

            serializedObject.ApplyModifiedProperties();
        }

        private void ImportPlyToAsset(GaussianSplatRenderer renderer, string plyFileName)
        {
            string filePath = Path.Combine(Application.streamingAssetsPath, "GaussianSplatting", plyFileName);
            if (!File.Exists(filePath))
            {
                EditorUtility.DisplayDialog("Error", $"File not found: {filePath}", "OK");
                return;
            }

            try
            {
                EditorUtility.DisplayProgressBar("Importing PLY", "Reading file...", 0.2f);
                byte[] fileBytes = File.ReadAllBytes(filePath);
                
                EditorUtility.DisplayProgressBar("Importing PLY", "Parsing...", 0.5f);
                var data = PlyGaussianSplatLoader.Load(fileBytes, CoordinateConversion.RightHandedToUnity);
                
                EditorUtility.DisplayProgressBar("Importing PLY", "Creating asset...", 0.8f);
                
                var asset = ScriptableObject.CreateInstance<GaussianSplatAsset>();
                asset.SetData(data);
                asset.SourcePath = filePath;
                
                Undo.RecordObject(renderer, "Import Gaussian Splat");
                renderer.Asset = asset;
                
                if (renderer.isActiveAndEnabled)
                    renderer.LoadAssetToGPU();
                
                EditorUtility.SetDirty(renderer);
                SceneView.RepaintAll();
                
                Debug.Log($"[GaussianSplatRenderer] Imported {plyFileName}: {data.Count} splats");
            }
            catch (Exception ex)
            {
                EditorUtility.DisplayDialog("Error", $"Failed to import: {ex.Message}", "OK");
                Debug.LogError($"[GaussianSplatRenderer] Import error: {ex}");
            }
            finally
            {
                EditorUtility.ClearProgressBar();
            }
        }
    }
}
#endif
