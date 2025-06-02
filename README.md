# XMAD-Bench: Cross-Domain Multilingual Audio Deepfake Benchmark
### by Ioan-Paul Ciobanu, Andrei-Iulian Hiji, Nicolae-Catalin Ristea, Paul Irofti, Cristian Rusu, Radu Tudor Ionescu



## License

The source code and models are released under the Creative Common Attribution-NonCommercial-ShareAlike 4.0 International ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)) license.

## Description
Recent advances in audio generation led to an increasing number of deepfakes, making the general public more vulnerable to financial scams, identity theft, and misinformation. Audio deepfake detectors promise to 
alleviate this issue, with many recent studies reporting accuracy rates close to $99\%$. However, these methods are typically tested in an in-domain setup, where the deepfake samples from the training and test sets 
are produced by the same generative models. To this end, we introduce XMAD-Bench, a large-scale cross-domain multilingual audio deepfake benchmark comprising 668.8 hours of real and deepfake speech. In our novel dataset, 
the speakers, the generative methods, and the real audio sources are distinct across training and test splits. This leads to a challenging cross-domain evaluation setup, where audio deepfake detectors can be tested in the wild. 
Our in-domain and cross-domain experiments indicate a clear disparity between the in-domain performance of deepfake detectors, which is usually as high as $100\%$, and the cross-domain performance of the same models, which is sometimes 
similar to random chance. Our benchmark highlights the need for the development of robust audio deepfake detectors, which maintain their generalization capacity across different languages, speakers, generative methods, and data sources.

## Detection framework
Modify the detection/config.json with the desired locations. Then run:
```bash
python detection/main.py
```


# Demo generation script

```bash
python demo_script.py \
  --sentence "Generarea unui exemplu de test a reusit." \
  --refs ref1.wav ref2.wav ref3.wav ... \
  --output synthesized_sample.wav
```

## Romanian datasets generation

#### VITS + FreeVC

```python
    # in vits_freevc.py you need to modify the model to:
    vc_freevc = TTS("voice_conversion_models/multilingual/vctk/freevc24")
    model_name = "tts_models/ro/cv/vits"
```

#### VITS + KNN-VC

```python
    # in vits_knnvc.py you need to modify the model to:
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
    model_name = "tts_models/ro/cv/vits"
```

#### VITS + OpenVoice

```python
    # in vits_openvoice.py you need to modify the model to:
    vc_openvoice = TTS("voice_conversion_models/multilingual/multi-dataset/openvoice_v2")
    model_name = "tts_models/ro/cv/vits"
```


## Arabic datasets generation

#### fairseq + FreeVC

```python
    # in fairseq_freevc.py you need to modify the model to:
    vc_freevc = TTS("voice_conversion_models/multilingual/vctk/freevc24")
    model_name = "tts_models/ara/fairseq/vits"
```

#### fairseq + KNN-VC

```python
    # in fairseq_knnvc.py you need to modify the model to:
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
    model_name = "tts_models/ara/fairseq/vits"
```


#### XTTSv2

```python
    # in xttsv2.py you need to modify the model to:
    language = "ar"
```


## Russian datasets generation

#### VITS + KNN-VC

```python
    # in vits_knnvc.py you need to modify the model to:
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
    model_name = "tts_models/rus/fairseq/vits"
```

#### VITS + OpenVoice

```python
    # in vits_openvoice.py you need to modify the model to:
    vc_openvoice = TTS("voice_conversion_models/multilingual/multi-dataset/openvoice_v2")
    model_name = "tts_models/rus/fairseq/vits"
```

#### XTTSv2

```python
    # in xttsv2.py you need to modify the model to:
    language = "ru"
```


## English datasets generation

#### VITS + KNN-VC

```python
    # in vits_knnvc.py you need to modify the model to:
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
    model_name = "tts_models/en/ljspeech/vits"
```

#### VITS + OpenVoice

```python
    # in vits_openvoice.py you need to modify the model to:
    vc_openvoice = TTS("voice_conversion_models/multilingual/multi-dataset/openvoice_v2")
    model_name = "tts_models/eng/fairseq/vits"
```

#### XTTSv2

```python
    # in xttsv2.py you need to modify the model to:
    language = "en"
```

## German datasets generation

#### VITS + KNN-VC

```python
    # in vits_knnvc.py you need to modify the model to:
    knn_vc = torch.hub.load('voice_conversion_models/multilingual/multi-dataset/knnvc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
    model_name = "tts_models/de/css10/vits-neon"
```

#### XTTSv2

```python
    # in xttsv2.py you need to modify the model to:
    language = "de"
```

## Spanish datasets generation

#### VITS + OpenVoice

```python
    # in vits_openvoice.py you need to modify the model to:
    vc_openvoice = TTS("voice_conversion_models/multilingual/multi-dataset/openvoice_v2")
    model_name = "tts_models/spa/fairseq/vits"
```

#### XTTSv2

```python
    # in xttsv2.py you need to modify the model to:
    language = "es"
```

## Mandarin datasets generation

#### Tacotron + KNNVC

```python
    # in vits_knnvc.py you need to modify the model to:
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
    model_name = "tts_models/zh-CN/baker/tacotron2-DDC-GST"
```

#### Bark + FreeVC

```python
    # in vits_freevc.py you need to modify the model to:
    vc_freevc = TTS("voice_conversion_models/multilingual/vctk/freevc24")
    model_name = "tts_models/multilingual/multi-dataset/bark"
```