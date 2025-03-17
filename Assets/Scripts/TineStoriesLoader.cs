using UnityEngine;
using Unity.Sentis;
using UnityEngine.Timeline;
public class TinyStoriesLoader : MonoBehaviour
{
    private Model runtimeModel;        // Модель Sentis
    public Worker worker;            // механизм логического вывода
    public ModelAsset modelAsset;
    void Start()
    {
        // Загрузить модель из файла ONNX
        if (modelAsset == null)
        {
            Debug.LogError("ONNX model file is not assigned.");
            return;
        }
        runtimeModel = ModelLoader.Load(modelAsset);
        Debug.Log("Model loaded successfully!");
        // Создать worker для выполнения модели
        worker = new Worker(runtimeModel, BackendType.CPU); // Для CPU
        Debug.Log("Worker created successfully!");
    }
    private void OnDestroy()
    {
        // Очистка ресурсов
        if (worker != null) worker.Dispose();
    }
}



