using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;
using System.IO;
using System.Text;
using FF = Unity.Sentis.Functional;
using TMPro;
using UnityEngine.UI;

public class RunTinyStories : MonoBehaviour
{
    const BackendType backend = BackendType.GPUCompute;

    string outputString = "";
    const int maxTokens = 100;
    const float predictability = 5f;
    const int END_OF_TEXT = 50256;

    string[] tokens;
    IWorker engine;

    int currentToken = 0;
    int[] outputTokens = new int[maxTokens];

    bool runInference = false;
    const int stopAfter = 100;
    int totalTokens = 0;

    string[] merges;
    Dictionary<string, int> vocab;

    public TMP_Text outputText;
    public TMP_InputField inputField;
    public Button generateButton;

    int[] whiteSpaceCharacters = new int[256];
    int[] encodedCharacters = new int[256];

    void Start()
    {
        generateButton.onClick.AddListener(OnGenerateButtonClicked);
        SetupWhiteSpaceShifts();
        LoadVocabulary();
    }

    void InitializeModel()
    {
        engine?.Dispose(); // Удаляем предыдущий движок, если он существует
        var model1 = ModelLoader.Load(Path.Join(Application.streamingAssetsPath, "tinystories.sentis"));
        var model2 = FF.Compile(
            (input, currentToken) =>
            {
                var row = FF.Select(model1.Forward(input)[8], 1, currentToken);
                return FF.Multinomial(predictability * row, 1);
            },
            (model1.inputs[0], InputDef.Int(new TensorShape()))
        );

        engine = WorkerFactory.CreateWorker(backend, model2);
    }

    void OnGenerateButtonClicked()
    {
        ResetGenerationState();  // Полностью сбрасываем состояние перед запуском новой генерации
        InitializeModel();       // Создаём новый экземпляр модели
        outputString = inputField.text;
        DecodePrompt(outputString);
        runInference = true;
    }

    void Update()
    {
        if (runInference)
        {
            RunInference();
        }
    }

    void RunInference()
    {
        using var tokensSoFar = new TensorInt(new TensorShape(1, maxTokens), outputTokens);
        using var index = new TensorInt(currentToken);

        engine.Execute(new Dictionary<string, Tensor> { { "input_0", tokensSoFar }, { "input_1", index } });

        var probs = engine.PeekOutput() as TensorInt;
        probs.CompleteOperationsAndDownload();

        int ID = probs[0];

        if (currentToken >= maxTokens - 1)
        {
            for (int i = 0; i < maxTokens - 1; i++) outputTokens[i] = outputTokens[i + 1];
            currentToken--;
        }

        outputTokens[++currentToken] = ID;
        totalTokens++;

        if (ID == END_OF_TEXT || totalTokens >= stopAfter)
        {
            runInference = false;
        }
        else
        {
            outputString += GetUnicodeText(tokens[ID]);
        }

        outputText.text = outputString;
    }

    void DecodePrompt(string text)
    {
        var inputTokens = GetTokens(text);

        for (int i = 0; i < inputTokens.Count; i++)
        {
            outputTokens[i] = inputTokens[i];
        }
        currentToken = inputTokens.Count - 1;
    }

    void LoadVocabulary()
    {
        var jsonText = File.ReadAllText(Path.Join(Application.streamingAssetsPath, "vocab.json"));
        vocab = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<string, int>>(jsonText);
        tokens = new string[vocab.Count];
        foreach (var item in vocab)
        {
            tokens[item.Value] = item.Key;
        }

        merges = File.ReadAllLines(Path.Join(Application.streamingAssetsPath, "merges.txt"));
    }

    string GetUnicodeText(string text)
    {
        var bytes = Encoding.GetEncoding("ISO-8859-1").GetBytes(ShiftCharacterDown(text));
        return Encoding.UTF8.GetString(bytes);
    }

    string GetASCIIText(string newText)
    {
        var bytes = Encoding.UTF8.GetBytes(newText);
        return ShiftCharacterUp(Encoding.GetEncoding("ISO-8859-1").GetString(bytes));
    }

    string ShiftCharacterDown(string text)
    {
        string outText = "";
        foreach (char letter in text)
        {
            outText += ((int)letter <= 256) ? letter :
                (char)whiteSpaceCharacters[(int)(letter - 256)];
        }
        return outText;
    }

    string ShiftCharacterUp(string text)
    {
        string outText = "";
        foreach (char letter in text)
        {
            outText += (char)encodedCharacters[(int)letter];
        }
        return outText;
    }

    void SetupWhiteSpaceShifts()
    {
        for (int i = 0, n = 0; i < 256; i++)
        {
            encodedCharacters[i] = i;
            if (IsWhiteSpace(i))
            {
                encodedCharacters[i] = n + 256;
                whiteSpaceCharacters[n++] = i;
            }
        }
    }

    bool IsWhiteSpace(int i)
    {
        return i <= 32 || (i >= 127 && i <= 160) || i == 173;
    }

    List<int> GetTokens(string text)
    {
        text = GetASCIIText(text);

        var inputTokens = new List<string>();
        foreach (var letter in text)
        {
            inputTokens.Add(letter.ToString());
        }

        ApplyMerges(inputTokens);

        var ids = new List<int>();
        foreach (var token in inputTokens)
        {
            if (vocab.TryGetValue(token, out int id))
            {
                ids.Add(id);
            }
        }

        return ids;
    }

    void ApplyMerges(List<string> inputTokens)
    {
        foreach (var merge in merges)
        {
            string[] pair = merge.Split(' ');
            int n = 0;
            while (n >= 0)
            {
                n = inputTokens.IndexOf(pair[0], n);
                if (n != -1 && n < inputTokens.Count - 1 && inputTokens[n + 1] == pair[1])
                {
                    inputTokens[n] += inputTokens[n + 1];
                    inputTokens.RemoveAt(n + 1);
                }
                if (n != -1) n++;
            }
        }
    }

    void ResetGenerationState()
    {
        if (engine != null)
        {
            engine.Dispose(); // Удаляем старый движок
        }

        currentToken = 0;
        totalTokens = 0;
        outputTokens = new int[maxTokens];
        outputString = "";
    }

    private void OnDestroy()
    {
        engine?.Dispose();
    }
}
