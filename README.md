# Demo generation script

```bash
python demo_script.py \
  --sentence "Generarea unui exemplu de test a reusit." \
  --refs ref1.wav ref2.wav ref3.wav ... \
  --output synthesized_sample.wav
```

# Romanian datasets generation

### VITS + FreeVC

```python
    # in vits_freevc.py you need to modify the model to:
    vc_freevc = TTS("voice_conversion_models/multilingual/vctk/freevc24")
    model_name = "tts_models/ro/cv/vits"
```

### VITS + KNN-VC

```python
    # in vits_knnvc.py you need to modify the model to:
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
    model_name = "tts_models/ro/cv/vits"
```

### VITS + OpenVoice

```python
    # in vits_openvoice.py you need to modify the model to:
    vc_openvoice = TTS("voice_conversion_models/multilingual/multi-dataset/openvoice_v2")
    model_name = "tts_models/ro/cv/vits"
```


# Arabic datasets generation

### fairseq + FreeVC

```python
    # in fairseq_freevc.py you need to modify the model to:
    vc_freevc = TTS("voice_conversion_models/multilingual/vctk/freevc24")
    model_name = "tts_models/ara/fairseq/vits"
```

### fairseq + KNN-VC

```python
    # in fairseq_knnvc.py you need to modify the model to:
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
    model_name = "tts_models/ara/fairseq/vits"
```


### XTTSv2

```python
    # in xttsv2.py you need to modify the model to:
    language = "ar"
```


# Russian datasets generation

### VITS + KNN-VC

```python
    # in vits_knnvc.py you need to modify the model to:
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
    model_name = "tts_models/rus/fairseq/vits"
```

### VITS + OpenVoice

```python
    # in vits_openvoice.py you need to modify the model to:
    vc_openvoice = TTS("voice_conversion_models/multilingual/multi-dataset/openvoice_v2")
    model_name = "tts_models/rus/fairseq/vits"
```

### XTTSv2

```python
    # in xttsv2.py you need to modify the model to:
    language = "ru"
```


# English datasets generation

### VITS + KNN-VC

```python
    # in vits_knnvc.py you need to modify the model to:
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
    model_name = "tts_models/en/ljspeech/vits"
```

### VITS + OpenVoice

```python
    # in vits_openvoice.py you need to modify the model to:
    vc_openvoice = TTS("voice_conversion_models/multilingual/multi-dataset/openvoice_v2")
    model_name = "tts_models/eng/fairseq/vits"
```

### XTTSv2

```python
    # in xttsv2.py you need to modify the model to:
    language = "en"
```