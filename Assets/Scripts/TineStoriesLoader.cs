using UnityEngine;
using Unity.Sentis;
using UnityEngine.Timeline;
public class TinyStoriesLoader : MonoBehaviour
{
    private Model runtimeModel;        // ������ Sentis
    public Worker worker;            // �������� ����������� ������
    public ModelAsset modelAsset;
    void Start()
    {
        // ��������� ������ �� ����� ONNX
        if (modelAsset == null)
        {
            Debug.LogError("ONNX model file is not assigned.");
            return;
        }
        runtimeModel = ModelLoader.Load(modelAsset);
        Debug.Log("Model loaded successfully!");
        // ������� worker ��� ���������� ������
        worker = new Worker(runtimeModel, BackendType.CPU); // ��� CPU
        Debug.Log("Worker created successfully!");
    }
    private void OnDestroy()
    {
        // ������� ��������
        if (worker != null) worker.Dispose();
    }
}



