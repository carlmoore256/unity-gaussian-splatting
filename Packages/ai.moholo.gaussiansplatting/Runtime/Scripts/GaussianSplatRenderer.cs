using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace GaussianSplatting
{
    public enum SortAlgorithm
    {
        Bitonic,
        Radix,
        None
    }

    [ExecuteAlways]
    public class GaussianSplatRenderer : MonoBehaviour, ISerializationCallbackReceiver
    {
        private static uint PackHalf2x16(float x, float y)
        {
            ushort hx = FloatToHalf(x);
            ushort hy = FloatToHalf(y);
            return (uint)hx | ((uint)hy << 16);
        }
        
        private static ushort FloatToHalf(float value)
        {
            int i = BitConverter.SingleToInt32Bits(value);
            int s = (i >> 16) & 0x8000;
            int e = ((i >> 23) & 0xff) - 127 + 15;
            int m = i & 0x7fffff;
            
            if (e <= 0)
            {
                if (e < -10) return (ushort)s;
                m = (m | 0x800000) >> (1 - e);
                return (ushort)(s | (m >> 13));
            }
            if (e == 0xff - 127 + 15)
            {
                if (m == 0) return (ushort)(s | 0x7c00);
                return (ushort)(s | 0x7c00 | (m >> 13));
            }
            if (e > 30)
            {
                return (ushort)(s | 0x7c00);
            }
            return (ushort)(s | (e << 10) | (m >> 13));
        }

        private static void ComputeCovariance3D(Vector4 rotation, Vector3 scale, out Vector3 covA, out Vector3 covB)
        {
            float qx = rotation.x, qy = rotation.y, qz = rotation.z, qw = rotation.w;
            float len = Mathf.Sqrt(qx*qx + qy*qy + qz*qz + qw*qw);
            if (len > 0.0001f) { qx /= len; qy /= len; qz /= len; qw /= len; }
            
            float r00 = 1f - 2f * (qy*qy + qz*qz);
            float r01 = 2f * (qx*qy - qw*qz);
            float r02 = 2f * (qx*qz + qw*qy);
            float r10 = 2f * (qx*qy + qw*qz);
            float r11 = 1f - 2f * (qx*qx + qz*qz);
            float r12 = 2f * (qy*qz - qw*qx);
            float r20 = 2f * (qx*qz - qw*qy);
            float r21 = 2f * (qy*qz + qw*qx);
            float r22 = 1f - 2f * (qx*qx + qy*qy);
            
            float m00 = r00 * scale.x, m01 = r01 * scale.y, m02 = r02 * scale.z;
            float m10 = r10 * scale.x, m11 = r11 * scale.y, m12 = r12 * scale.z;
            float m20 = r20 * scale.x, m21 = r21 * scale.y, m22 = r22 * scale.z;
            
            float v00 = m00*m00 + m01*m01 + m02*m02;
            float v01 = m00*m10 + m01*m11 + m02*m12;
            float v02 = m00*m20 + m01*m21 + m02*m22;
            float v11 = m10*m10 + m11*m11 + m12*m12;
            float v12 = m10*m20 + m11*m21 + m12*m22;
            float v22 = m20*m20 + m21*m21 + m22*m22;
            
            covA = new Vector3(v00, v01, v02);
            covB = new Vector3(v11, v12, v22);
        }

        [Header("Rendering")]
        public Material Material;
        public Camera TargetCamera;

        [Header("Splat Properties")]
        [Range(0.1f, 5.0f)]
        public float ScaleMultiplier = 1.0f;
        [Min(0)]
        public int MaxSplatsToLoad = 0;
        
        [Header("Performance")]
        [Range(1, 90)]
        public int SortEveryNFrames = 1;
        public bool UseRenderFeature = false;
        public bool EnableFrustumCulling = true;
        [Range(0.0f, 1.0f)]
        public float FrustumCullMargin = 0.3f;
        public SortAlgorithm SortingAlgorithm = SortAlgorithm.Bitonic;
        
        [HideInInspector]
        public int LogPerformanceEveryNFrames = 0;

        [Header("Asset Reference (Optional)")]
        [Tooltip("Pre-imported GaussianSplatAsset. If set, will auto-load on enable.")]
        [SerializeField] private GaussianSplatAsset _asset;
        
        public GaussianSplatAsset Asset
        {
            get => _asset;
            set
            {
                _asset = value;
                if (_asset != null && isActiveAndEnabled)
                    LoadAssetToGPU();
            }
        }

        private GraphicsBuffer _glesPosScale;
        private GraphicsBuffer _glesRotation;
        private GraphicsBuffer _glesColor;

        private uint[] _orderCpu = Array.Empty<uint>();
        private Vector3[] _centersCpu = Array.Empty<Vector3>();
        private int _count;

        private Bounds _localBounds;
        private Bounds _worldBounds;

        private GaussianMeshDeformer _meshDeformer;
        private MaterialPropertyBlock _mpb;
        private Material _activeMaterial;
        
        private ComputeShader _precomputeShader;
        private int _precomputeKernel;
        private const int GPU_PRECOMPUTE_THRESHOLD = 50000;
        
        private int _frameCounter = 0;
        
        private class CameraRenderState
        {
            public GraphicsBuffer OrderBuffer;
            public GaussianSplatBitonicSorter BitonicSorter;
            public GaussianSplatRadixSorter RadixSorter;
            public int VisibleCount;
            public int LastFrameUsed;
            
            public void Dispose()
            {
                OrderBuffer?.Release();
                BitonicSorter?.Dispose();
                RadixSorter?.Dispose();
            }
        }
        private Dictionary<Camera, CameraRenderState> _cameraStates = new Dictionary<Camera, CameraRenderState>();
        private const int CAMERA_STATE_EXPIRY_FRAMES = 300;
        
        private float _lastSortTimeMs = 0f;
        private float _avgSortTimeMs = 0f;
        private float _avgFrameTimeMs = 0f;
        private int _perfLogCounter = 0;
        private System.Diagnostics.Stopwatch _sortStopwatch = new System.Diagnostics.Stopwatch();
        private float _lastFrameTime = 0f;
        private bool _firstRenderLogged = false;

        public bool IsLoaded => _count > 0 && _glesPosScale != null;
        public int SplatCount => _count;
        public GraphicsBuffer PositionBuffer => _glesPosScale;
        public bool IsMeshDeformationEnabled => _meshDeformer != null && _meshDeformer.IsInitialized;

        private void OnEnable()
        {
            RenderPipelineManager.beginCameraRendering += BeginCameraRendering;
            if (_mpb == null) _mpb = new MaterialPropertyBlock();
            
            if (_asset != null && !IsLoaded)
                LoadAssetToGPU();
        }

        private void OnDisable()
        {
            RenderPipelineManager.beginCameraRendering -= BeginCameraRendering;
            if (!Application.isPlaying)
                ReleaseBuffersOnly();
            else
                Release();
        }

        private void OnDestroy()
        {
            Release();
        }

        private void ReleaseBuffersOnly()
        {
            foreach (var state in _cameraStates.Values)
                state.Dispose();
            _cameraStates.Clear();
            
            _meshDeformer?.Dispose(); _meshDeformer = null;
            _glesPosScale?.Release(); _glesPosScale = null;
            _glesRotation?.Release(); _glesRotation = null;
            _glesColor?.Release(); _glesColor = null;
        }

        private void Release()
        {
            foreach (var state in _cameraStates.Values)
                state.Dispose();
            _cameraStates.Clear();
            
            _meshDeformer?.Dispose(); _meshDeformer = null;
            _glesPosScale?.Release(); _glesPosScale = null;
            _glesRotation?.Release(); _glesRotation = null;
            _glesColor?.Release(); _glesColor = null;
            
            _orderCpu = Array.Empty<uint>();
            _centersCpu = Array.Empty<Vector3>();
            _count = 0;
            _localBounds = new Bounds(Vector3.zero, Vector3.zero);
            _worldBounds = new Bounds(Vector3.zero, Vector3.zero);
        }

        public void SetSplatData(GaussianSplatData data)
        {
            if (data == null || data.Count == 0)
            {
                Debug.LogWarning("[GaussianSplatRenderer] SetSplatData: data is null or empty");
                return;
            }

            if (Material == null)
            {
                Debug.LogWarning("[GaussianSplatRenderer] SetSplatData: Material is null");
                return;
            }

            _activeMaterial = Material;
            Release();

            int count = data.Count;
            Vector3[] centers = data.Centers;
            Vector4[] rotations = data.Rotations;
            Vector3[] scales = data.Scales;
            Vector4[] colors = data.Colors;

            if (MaxSplatsToLoad > 0 && MaxSplatsToLoad < count)
            {
                count = MaxSplatsToLoad;
                Array.Resize(ref centers, count);
                Array.Resize(ref rotations, count);
                Array.Resize(ref scales, count);
                Array.Resize(ref colors, count);
            }

            _count = count;
            _centersCpu = centers;
            _orderCpu = new uint[_count];
            for (uint i = 0; i < _orderCpu.Length; i++) _orderCpu[i] = i;

            for (int i = 0; i < scales.Length; i++)
                scales[i] = scales[i] * ScaleMultiplier;

            _glesPosScale = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _count, sizeof(float) * 4);
            _glesRotation = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _count, sizeof(float) * 4);
            _glesColor = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _count, sizeof(float) * 4);

            bool useGpuPrecompute = SystemInfo.supportsComputeShaders && _count >= GPU_PRECOMPUTE_THRESHOLD;

            if (useGpuPrecompute)
                PrecomputeCovarianceGPU(centers, rotations, scales, colors);
            else
                PrecomputeCovarianceCPU(centers, rotations, scales, colors);

            _localBounds = ComputeLocalBounds(_centersCpu);
            _worldBounds = TransformBounds(_localBounds, transform.localToWorldMatrix);
            _frameCounter = 0;

            Debug.Log($"[GaussianSplatRenderer] SetSplatData: loaded {_count} splats");
        }

        private CameraRenderState GetOrCreateCameraState(Camera camera)
        {
            if (_cameraStates.TryGetValue(camera, out var state))
            {
                state.LastFrameUsed = Time.frameCount;
                return state;
            }
            
            state = new CameraRenderState
            {
                OrderBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.CopyDestination, _count, sizeof(uint)),
                VisibleCount = _count,
                LastFrameUsed = Time.frameCount
            };
            
            state.OrderBuffer.SetData(_orderCpu);
            
            if (SortingAlgorithm == SortAlgorithm.Bitonic)
            {
                var sortShader = Resources.Load<ComputeShader>("GaussianSplatBitonicSort");
                if (sortShader != null)
                    state.BitonicSorter = new GaussianSplatBitonicSorter(sortShader, _count);
            }
            else if (SortingAlgorithm == SortAlgorithm.Radix)
            {
                var sortShader = Resources.Load<ComputeShader>("GaussianSplatRadixSort");
                if (sortShader != null)
                    state.RadixSorter = new GaussianSplatRadixSorter(sortShader, _count);
            }
            
            _cameraStates[camera] = state;
            CleanupUnusedCameraStates();
            
            return state;
        }
        
        private void CleanupUnusedCameraStates()
        {
            var toRemove = new List<Camera>();
            int currentFrame = Time.frameCount;
            
            foreach (var kvp in _cameraStates)
            {
                if (kvp.Key == null || (currentFrame - kvp.Value.LastFrameUsed) > CAMERA_STATE_EXPIRY_FRAMES)
                {
                    kvp.Value.Dispose();
                    toRemove.Add(kvp.Key);
                }
            }
            
            foreach (var cam in toRemove)
                _cameraStates.Remove(cam);
        }

        public bool InitializeMeshDeformation(GaussianFaceMapping[] mappings, Vector3[] vertices, int[] triangles)
        {
            if (_count == 0 || _glesPosScale == null)
            {
                Debug.LogError("[GaussianSplatRenderer] Cannot initialize mesh deformation - Gaussians not loaded");
                return false;
            }
            
            if (mappings.Length != _count)
            {
                Debug.LogError($"[GaussianSplatRenderer] Mapping count ({mappings.Length}) doesn't match Gaussian count ({_count})");
                return false;
            }
            
            int faceCount = triangles.Length / 3;
            
            var deformShader = Resources.Load<ComputeShader>("GaussianMeshDeform");
            if (deformShader == null)
            {
                Debug.LogError("[GaussianSplatRenderer] Could not load GaussianMeshDeform compute shader");
                return false;
            }
            
            _meshDeformer?.Dispose();
            _meshDeformer = new GaussianMeshDeformer(deformShader, _count, faceCount);
            _meshDeformer.Initialize(mappings, vertices, triangles);
            _meshDeformer.StoreOriginalCovariances(_glesPosScale, _glesRotation, _glesColor);
            
            Debug.Log($"[GaussianSplatRenderer] Mesh deformation initialized for {_count} Gaussians, {faceCount} faces");
            return true;
        }

        public void UpdateMeshDeformation(Vector3[] vertices, int[] triangles)
        {
            if (_meshDeformer == null || !_meshDeformer.IsInitialized)
            {
                Debug.LogWarning("[GaussianSplatRenderer] Mesh deformation not initialized");
                return;
            }
            
            _meshDeformer.UpdateMeshState(vertices, triangles);
        }

        private void PrecomputeCovarianceGPU(Vector3[] centers, Vector4[] rotations, Vector3[] scales, Vector4[] colors)
        {
            var stopwatch = new System.Diagnostics.Stopwatch();
            stopwatch.Start();
            
            if (_precomputeShader == null)
            {
                _precomputeShader = Resources.Load<ComputeShader>("GaussianSplatPrecompute");
                if (_precomputeShader == null)
                {
                    Debug.LogWarning("[GaussianSplatRenderer] Could not load GaussianSplatPrecompute compute shader. Falling back to CPU.");
                    PrecomputeCovarianceCPU(centers, rotations, scales, colors);
                    return;
                }
                _precomputeKernel = _precomputeShader.FindKernel("CSPrecomputeCovariance");
            }
            
            var centersBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _count, sizeof(float) * 3);
            var rotationsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _count, sizeof(float) * 4);
            var scalesBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _count, sizeof(float) * 3);
            var colorsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _count, sizeof(float) * 4);
            
            centersBuffer.SetData(centers);
            rotationsBuffer.SetData(rotations);
            scalesBuffer.SetData(scales);
            colorsBuffer.SetData(colors);
            
            _precomputeShader.SetBuffer(_precomputeKernel, "_Centers", centersBuffer);
            _precomputeShader.SetBuffer(_precomputeKernel, "_Rotations", rotationsBuffer);
            _precomputeShader.SetBuffer(_precomputeKernel, "_Scales", scalesBuffer);
            _precomputeShader.SetBuffer(_precomputeKernel, "_Colors", colorsBuffer);
            _precomputeShader.SetBuffer(_precomputeKernel, "_OutPosCovA", _glesPosScale);
            _precomputeShader.SetBuffer(_precomputeKernel, "_OutCovB", _glesRotation);
            _precomputeShader.SetBuffer(_precomputeKernel, "_OutCovCColor", _glesColor);
            _precomputeShader.SetInt("_SplatCount", _count);
            
            int threadGroups = (_count + 255) / 256;
            _precomputeShader.Dispatch(_precomputeKernel, threadGroups, 1, 1);
            
            centersBuffer.Release();
            rotationsBuffer.Release();
            scalesBuffer.Release();
            colorsBuffer.Release();
            
            stopwatch.Stop();
            Debug.Log($"[GaussianSplatRenderer] GPU covariance precompute: {_count} splats in {stopwatch.ElapsedMilliseconds}ms");
        }

        private void PrecomputeCovarianceCPU(Vector3[] centers, Vector4[] rotations, Vector3[] scales, Vector4[] colors)
        {
            var stopwatch = new System.Diagnostics.Stopwatch();
            stopwatch.Start();
            
            Vector4[] posCovA = new Vector4[_count];
            Vector4[] covB = new Vector4[_count];
            Vector4[] covCColor = new Vector4[_count];
            
            for (int i = 0; i < _count; i++)
            {
                ComputeCovariance3D(rotations[i], scales[i], out Vector3 covA, out Vector3 covBVec);
                
                posCovA[i] = new Vector4(centers[i].x, centers[i].y, centers[i].z, covA.x);
                covB[i] = new Vector4(covA.y, covA.z, covBVec.x, covBVec.y);
                
                uint colorRG = PackHalf2x16(colors[i].x, colors[i].y);
                uint colorBA = PackHalf2x16(colors[i].z, colors[i].w);
                covCColor[i] = new Vector4(covBVec.z, 0f, 
                    BitConverter.Int32BitsToSingle((int)colorRG),
                    BitConverter.Int32BitsToSingle((int)colorBA));
            }
            
            _glesPosScale.SetData(posCovA);
            _glesRotation.SetData(covB);
            _glesColor.SetData(covCColor);
            
            stopwatch.Stop();
            Debug.Log($"[GaussianSplatRenderer] CPU covariance precompute: {_count} splats in {stopwatch.ElapsedMilliseconds}ms");
        }

        private void BeginCameraRendering(ScriptableRenderContext context, Camera camera)
        {
            if (UseRenderFeature && GraphicsSettings.currentRenderPipeline != null)
                return;
            
            if (!enabled || _activeMaterial == null || _glesPosScale == null || _count == 0)
                return;
            
            if (camera.cameraType != CameraType.SceneView)
            {
                var cam = TargetCamera != null ? TargetCamera : Camera.main;
                if (cam != null && camera != cam)
                    return;
            }

            RenderForCamera(camera, (CommandBuffer)null);
        }

        private void OnRenderObject()
        {
            if (GraphicsSettings.currentRenderPipeline != null) return;
            if (!enabled || _activeMaterial == null || _glesPosScale == null || _count == 0) return;

            var camera = Camera.current;
            if (camera == null) return;

            var cam = TargetCamera != null ? TargetCamera : Camera.main;
            if (cam != null && camera != cam) return;

            RenderForCamera(camera, (CommandBuffer)null);
        }

        public void RenderWithCommandBuffer(CommandBuffer cmd, Camera camera)
        {
            if (!enabled || _activeMaterial == null || _glesPosScale == null || _count == 0) return;
            
            if (camera.cameraType != CameraType.SceneView)
            {
                var cam = TargetCamera != null ? TargetCamera : Camera.main;
                if (cam != null && camera != cam) return;
            }

            RenderForCamera(camera, cmd);
        }

        public void RenderWithRasterCommandBuffer(RasterCommandBuffer cmd, Camera camera)
        {
            if (!enabled || _activeMaterial == null || _glesPosScale == null || _count == 0) return;
            
            if (camera.cameraType != CameraType.SceneView)
            {
                var cam = TargetCamera != null ? TargetCamera : Camera.main;
                if (cam != null && camera != cam) return;
            }

            RenderForCamera(camera, cmd);
        }

        private void RenderForCamera(Camera camera, CommandBuffer cmd)
        {
            if (!_firstRenderLogged)
            {
                _firstRenderLogged = true;
                Debug.Log($"[GaussianSplatRenderer] First render: Camera={camera.name}");
            }
            
            var camState = GetOrCreateCameraState(camera);
            
            float currentTime = Time.realtimeSinceStartup;
            float frameTime = (currentTime - _lastFrameTime) * 1000f;
            _lastFrameTime = currentTime;
            _avgFrameTimeMs = Mathf.Lerp(_avgFrameTimeMs, frameTime, 0.1f);
            
            _frameCounter++;
            bool shouldSort = (_frameCounter % SortEveryNFrames) == 0;
            
            if (shouldSort)
            {
                _sortStopwatch.Restart();
                
                if (_meshDeformer != null && _meshDeformer.IsInitialized)
                    _meshDeformer.ApplyDeformation(_glesPosScale, _glesRotation, _glesColor);
                
                var camPosOS = transform.InverseTransformPoint(camera.transform.position);
                var camDirOS = transform.InverseTransformDirection(camera.transform.forward).normalized;

                if (SortingAlgorithm == SortAlgorithm.None)
                {
                    camState.VisibleCount = _count;
                }
                else if (camState.BitonicSorter != null)
                {
                    Matrix4x4 sortModelMatrix = transform.localToWorldMatrix;
                    Matrix4x4 sortViewMatrix = camera.worldToCameraMatrix;
                    Matrix4x4 modelViewMatrix = sortViewMatrix * sortModelMatrix;
                    
                    if (EnableFrustumCulling)
                    {
                        Matrix4x4 viewProjMatrix = camera.projectionMatrix * sortViewMatrix * sortModelMatrix;
                        if (camState.BitonicSorter.Sort(_glesPosScale, camState.OrderBuffer, modelViewMatrix, viewProjMatrix, camPosOS, camDirOS, FrustumCullMargin, out int visible))
                            camState.VisibleCount = visible;
                    }
                    else
                    {
                        camState.BitonicSorter.Sort(_glesPosScale, camState.OrderBuffer, modelViewMatrix, camPosOS, camDirOS);
                        camState.VisibleCount = _count;
                    }
                }
                else if (camState.RadixSorter != null)
                {
                    Matrix4x4 sortModelMatrix = transform.localToWorldMatrix;
                    Matrix4x4 sortViewMatrix = camera.worldToCameraMatrix;
                    Matrix4x4 modelViewMatrix = sortViewMatrix * sortModelMatrix;
                    
                    if (EnableFrustumCulling)
                    {
                        Matrix4x4 viewProjMatrix = camera.projectionMatrix * sortViewMatrix * sortModelMatrix;
                        if (camState.RadixSorter.Sort(_glesPosScale, camState.OrderBuffer, modelViewMatrix, viewProjMatrix, camPosOS, camDirOS, FrustumCullMargin, out int visible))
                            camState.VisibleCount = visible;
                    }
                    else
                    {
                        camState.RadixSorter.Sort(_glesPosScale, camState.OrderBuffer, modelViewMatrix, camPosOS, camDirOS);
                        camState.VisibleCount = _count;
                    }
                }
                
                _sortStopwatch.Stop();
                _lastSortTimeMs = (float)_sortStopwatch.Elapsed.TotalMilliseconds;
                _avgSortTimeMs = Mathf.Lerp(_avgSortTimeMs, _lastSortTimeMs, 0.1f);
            }
            
            int visibleCount = _count;
            int reportedVisible = camState.VisibleCount > 0 ? camState.VisibleCount : _count;
            
            if (LogPerformanceEveryNFrames > 0)
            {
                _perfLogCounter++;
                if (_perfLogCounter >= LogPerformanceEveryNFrames)
                {
                    _perfLogCounter = 0;
                    float fps = _avgFrameTimeMs > 0 ? 1000f / _avgFrameTimeMs : 0;
                    float cullPercent = _count > 0 ? (1f - (float)reportedVisible / _count) * 100f : 0;
                    Debug.Log($"[GS-PERF] FPS: {fps:F1} | Sort: {_avgSortTimeMs:F2}ms | Visible: {reportedVisible}/{_count} ({cullPercent:F0}% culled)");
                }
            }

            float w = camera.pixelWidth;
            float h = camera.pixelHeight;
            var viewportSize = new Vector4(w, h, 1f / Mathf.Max(1f, w), 1f / Mathf.Max(1f, h));

            _activeMaterial.SetBuffer("_SplatOrder", camState.OrderBuffer);
            _activeMaterial.SetBuffer("_SplatPosCovA", _glesPosScale);
            _activeMaterial.SetBuffer("_SplatCovB", _glesRotation);
            _activeMaterial.SetBuffer("_SplatCovCColor", _glesColor);

            if (cmd != null)
            {
                _mpb.Clear();
                _mpb.SetVector("_ViewportSize", viewportSize);
                _mpb.SetFloat("_IsOrtho", camera.orthographic ? 1f : 0f);
                _mpb.SetInt("_NumSplats", visibleCount);
                _mpb.SetMatrix("_SplatObjectToWorld", transform.localToWorldMatrix);
                _mpb.SetMatrix("_SplatWorldToObject", transform.worldToLocalMatrix);
            }
            else
            {
                _activeMaterial.SetVector("_ViewportSize", viewportSize);
                _activeMaterial.SetFloat("_IsOrtho", camera.orthographic ? 1f : 0f);
                _activeMaterial.SetInt("_NumSplats", visibleCount);
                _activeMaterial.SetMatrix("_SplatObjectToWorld", transform.localToWorldMatrix);
                _activeMaterial.SetMatrix("_SplatWorldToObject", transform.worldToLocalMatrix);
            }

            var bounds = _localBounds.size.sqrMagnitude > 0 
                ? TransformBounds(_localBounds, transform.localToWorldMatrix) 
                : new Bounds(transform.position, Vector3.one * 100000f);
            
            if (cmd != null)
            {
                cmd.DrawProcedural(Matrix4x4.identity, _activeMaterial, 0, MeshTopology.Triangles, visibleCount * 6, 1, _mpb);
            }
            else if (GraphicsSettings.currentRenderPipeline == null)
            {
                _activeMaterial.SetPass(0);
                Graphics.DrawProceduralNow(MeshTopology.Triangles, visibleCount * 6, 1);
            }
            else
            {
                Graphics.DrawProcedural(_activeMaterial, bounds, MeshTopology.Triangles, visibleCount * 6, 1, camera);
            }
        }

        private void RenderForCamera(Camera camera, RasterCommandBuffer cmd)
        {
            if (!_firstRenderLogged)
            {
                _firstRenderLogged = true;
                Debug.Log($"[GaussianSplatRenderer] First render (RasterCB): Camera={camera.name}");
            }
            
            var camState = GetOrCreateCameraState(camera);
            
            float currentTime = Time.realtimeSinceStartup;
            float frameTime = (currentTime - _lastFrameTime) * 1000f;
            _lastFrameTime = currentTime;
            _avgFrameTimeMs = Mathf.Lerp(_avgFrameTimeMs, frameTime, 0.1f);
            
            _frameCounter++;
            bool shouldSort = (_frameCounter % SortEveryNFrames) == 0;
            
            if (shouldSort)
            {
                if (_meshDeformer != null && _meshDeformer.IsInitialized)
                    _meshDeformer.ApplyDeformation(_glesPosScale, _glesRotation, _glesColor);
                
                var camPosOS = transform.InverseTransformPoint(camera.transform.position);
                var camDirOS = transform.InverseTransformDirection(camera.transform.forward).normalized;

                if (SortingAlgorithm == SortAlgorithm.None)
                {
                    camState.VisibleCount = _count;
                }
                else if (camState.BitonicSorter != null)
                {
                    Matrix4x4 sortModelMatrix = transform.localToWorldMatrix;
                    Matrix4x4 sortViewMatrix = camera.worldToCameraMatrix;
                    Matrix4x4 modelViewMatrix = sortViewMatrix * sortModelMatrix;
                    
                    if (EnableFrustumCulling)
                    {
                        Matrix4x4 viewProjMatrix = camera.projectionMatrix * sortViewMatrix * sortModelMatrix;
                        if (camState.BitonicSorter.Sort(_glesPosScale, camState.OrderBuffer, modelViewMatrix, viewProjMatrix, camPosOS, camDirOS, FrustumCullMargin, out int visible))
                            camState.VisibleCount = visible;
                    }
                    else
                    {
                        camState.BitonicSorter.Sort(_glesPosScale, camState.OrderBuffer, modelViewMatrix, camPosOS, camDirOS);
                        camState.VisibleCount = _count;
                    }
                }
                else if (camState.RadixSorter != null)
                {
                    Matrix4x4 sortModelMatrix = transform.localToWorldMatrix;
                    Matrix4x4 sortViewMatrix = camera.worldToCameraMatrix;
                    Matrix4x4 modelViewMatrix = sortViewMatrix * sortModelMatrix;
                    
                    if (EnableFrustumCulling)
                    {
                        Matrix4x4 viewProjMatrix = camera.projectionMatrix * sortViewMatrix * sortModelMatrix;
                        if (camState.RadixSorter.Sort(_glesPosScale, camState.OrderBuffer, modelViewMatrix, viewProjMatrix, camPosOS, camDirOS, FrustumCullMargin, out int visible))
                            camState.VisibleCount = visible;
                    }
                    else
                    {
                        camState.RadixSorter.Sort(_glesPosScale, camState.OrderBuffer, modelViewMatrix, camPosOS, camDirOS);
                        camState.VisibleCount = _count;
                    }
                }
            }

            int visibleCount = _count;

            float w = camera.pixelWidth;
            float h = camera.pixelHeight;
            var viewportSize = new Vector4(w, h, 1f / Mathf.Max(1f, w), 1f / Mathf.Max(1f, h));

            _activeMaterial.SetBuffer("_SplatOrder", camState.OrderBuffer);
            _activeMaterial.SetBuffer("_SplatPosCovA", _glesPosScale);
            _activeMaterial.SetBuffer("_SplatCovB", _glesRotation);
            _activeMaterial.SetBuffer("_SplatCovCColor", _glesColor);
            
            _mpb.Clear();
            _mpb.SetVector("_ViewportSize", viewportSize);
            _mpb.SetFloat("_IsOrtho", camera.orthographic ? 1f : 0f);
            _mpb.SetInt("_NumSplats", visibleCount);
            _mpb.SetMatrix("_SplatObjectToWorld", transform.localToWorldMatrix);
            _mpb.SetMatrix("_SplatWorldToObject", transform.worldToLocalMatrix);

            var viewMatrix = camera.worldToCameraMatrix;
            var projMatrix = camera.projectionMatrix;
            cmd.SetViewProjectionMatrices(viewMatrix, projMatrix);
            
            _mpb.SetFloat("_CamProjM00", projMatrix[0, 0]);
            _mpb.SetFloat("_CamProjM11", projMatrix[1, 1]);

            cmd.DrawProcedural(Matrix4x4.identity, _activeMaterial, 0, MeshTopology.Triangles, visibleCount * 6, 1, _mpb);
        }

        private void OnDrawGizmosSelected()
        {
            if (_localBounds.size.sqrMagnitude <= 0) return;

            Gizmos.color = Color.yellow;
            var b = TransformBounds(_localBounds, transform.localToWorldMatrix);
            Gizmos.DrawWireCube(b.center, b.size);

            if (_centersCpu != null && _centersCpu.Length > 0)
            {
                Gizmos.color = Color.magenta;
                Gizmos.DrawSphere(transform.TransformPoint(_centersCpu[0]), 0.02f);
            }
        }

        private static Bounds ComputeLocalBounds(Vector3[] centers)
        {
            if (centers == null || centers.Length == 0)
                return new Bounds(Vector3.zero, Vector3.zero);

            var min = centers[0];
            var max = centers[0];
            for (int i = 1; i < centers.Length; i++)
            {
                var c = centers[i];
                min = Vector3.Min(min, c);
                max = Vector3.Max(max, c);
            }

            var b = new Bounds((min + max) * 0.5f, (max - min));
            b.Expand(Mathf.Max(0.01f, b.size.magnitude * 0.02f));
            return b;
        }

        private static Bounds TransformBounds(Bounds localBounds, Matrix4x4 localToWorld)
        {
            var center = localToWorld.MultiplyPoint3x4(localBounds.center);
            var ext = localBounds.extents;
            var axisX = localToWorld.MultiplyVector(new Vector3(ext.x, 0, 0));
            var axisY = localToWorld.MultiplyVector(new Vector3(0, ext.y, 0));
            var axisZ = localToWorld.MultiplyVector(new Vector3(0, 0, ext.z));
            var worldExt = new Vector3(
                Mathf.Abs(axisX.x) + Mathf.Abs(axisY.x) + Mathf.Abs(axisZ.x),
                Mathf.Abs(axisX.y) + Mathf.Abs(axisY.y) + Mathf.Abs(axisZ.y),
                Mathf.Abs(axisX.z) + Mathf.Abs(axisY.z) + Mathf.Abs(axisZ.z)
            );
            return new Bounds(center, worldExt * 2f);
        }

        public void OnBeforeSerialize() { }

        public void OnAfterDeserialize() { }

        public void LoadAssetToGPU()
        {
            if (_asset == null || _asset.Data == null)
            {
                Debug.LogWarning("[GaussianSplatRenderer] LoadAssetToGPU: No asset or asset data");
                return;
            }
            SetSplatData(_asset.Data);
        }
    }
}
