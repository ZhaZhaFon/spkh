
### 功能 Features

* reciple supported
    * LibriMix: for Speaker Extraction (SEx), re-implemented models including TD-SpeakerBeam...

### 文件组织 Files
```
    data/
    egs/ # recipes
        librimix/ # recipe LibriMix
            Makefile # guidance of HOW-TO use this recipe
            TD-SpeakerBeam.py
            TD-SpeakerBeam.yml
            TD-SpeakerBeam_eval.py
    model/
        speakerbeam/ # from SpeechFIT@BUT
            system.py
            td_speakerbeam.py
            base_model_informed.py
            adapt_layers.py
            LICENSE.txt
    Makefile

```