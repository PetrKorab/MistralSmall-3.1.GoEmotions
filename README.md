# Mistral-Small-3.1-24B-GoEmotions

[![Hugging Face](https://img.shields.io/badge/HuggingFace-TextMiningStories%2FMistral--Small--3.1--24B--goemotions-yellow?logo=huggingface)](https://huggingface.co/TextMiningStories/Mistral-Small-3.1-24B-goemotions)
[![Language](https://img.shields.io/badge/language-English-blue)]()
[![Dataset](https://img.shields.io/badge/dataset-GoEmotions-green)](https://huggingface.co/datasets/go_emotions)

Multi-label emotion classifier for short English text — detects up to **28 fine-grained emotions** simultaneously. Fine-tuned on [GoEmotions](https://huggingface.co/datasets/go_emotions) using [Mistral Small 3.1 24B](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) with LoRA. 26 of 28 emotion labels achieve F1 > 0.50 on the held-out test set.

---

## Quick Start

### 1 — Install dependencies

```bash
pip install unsloth peft transformers torch tensorflow tensorflow-hub huggingface_hub
```

### 2 — Download the model

```python
from huggingface_hub import snapshot_download

model_dir = snapshot_download(
    repo_id="TextMiningStories/Mistral-Small-3.1-24B-goemotions",
    local_dir=r"C:\path\to\goemotions_model",   # use local_dir on Windows to avoid symlink errors
)
```

### 3 — Run inference

```python
import sys
sys.path.insert(0, model_dir)
from infer import EmotionClassifier

clf = EmotionClassifier(model_dir)

texts = [
    "I can't believe how amazing that was!",
    "This is outrageous, I am absolutely furious.",
    "I feel a bit anxious about tomorrow.",
]

for text, pred in zip(texts, clf.predict(texts)):
    print(f"{text}")
    print(f"  → {pred}\n")
```

**Example output:**
```
I can't believe how amazing that was!
  → {'admiration': 0.9123, 'excitement': 0.8741, 'surprise': 0.7056}

This is outrageous, I am absolutely furious.
  → {'anger': 0.9388, 'annoyance': 0.8112, 'disapproval': 0.7045}

I feel a bit anxious about tomorrow.
  → {'nervousness': 0.9512, 'fear': 0.6834}
```

---

## Architecture

Text is encoded with **Universal Sentence Encoder v4** (512-dim), projected into the Mistral hidden space via a two-layer MLP, passed through a single-token sequence of the LoRA-adapted **Mistral-Small-3.1-24B** transformer, and classified with a linear head over 28 labels. Sigmoid + threshold 0.50 produces the multi-hot output.

```
Raw text  →  USE v4 (512-dim)  →  MLP (512→2560→5120)
         →  Mistral-24B LoRA (single token, last hidden state)
         →  Dropout + Linear (5120→28)  →  sigmoid @ 0.50
```

**Training:** 15 epochs · LoRA r=16 · 4-bit quantised · focal loss · ~18 h on RTX 6000 Ada (48 GB)

---

## Results

Evaluated on 7,891 held-out English samples at threshold = 0.50.

| Emotion | F1 | | Emotion | F1 |
|---|---|---|---|---|
| desire | 0.929 | | sadness | 0.871 |
| caring | 0.894 | | surprise | 0.851 |
| disgust | 0.886 | | disappointment | 0.847 |
| realization | 0.881 | | confusion | 0.790 |
| gratitude | 0.880 | | optimism | 0.788 |
| excitement | 0.877 | | joy | 0.779 |

26 of 28 labels exceed F1 = 0.50. Only **approval** and **neutral** fall below due to heavy semantic overlap with other classes.

---

## Warning — Low-Support Labels

> The following 6 labels each have **fewer than 100 positive test samples**. Their high F1 scores are likely inflated by the small sample size and are **not reliable for production use**:
>
> `relief` · `embarrassment` · `nervousness` · `pride` · `remorse` · `grief`

To suppress them from predictions:

```python
UNRELIABLE = {"relief", "embarrassment", "nervousness", "pride", "remorse", "grief"}
safe = [{k: v for k, v in pred.items() if k not in UNRELIABLE} for pred in clf.predict(texts)]
```

---

## Repository Contents

| File | Purpose |
|---|---|
| `funetuning.ipynb` | Full training pipeline — data loading, model setup, LoRA fine-tuning, evaluation |
| `inference.ipynb` | Download from Hugging Face and run inference (no local model dir needed) 

---

## Requirements

```
unsloth>=2026.5.2
peft>=0.18.1
transformers>=4.57.6
torch>=2.10.0
tensorflow>=2.0
tensorflow-hub>=0.12
huggingface_hub
numpy
```

GPU with **48 GB VRAM** required for 4-bit inference. CPU inference is not practical.

---

## Links

- **Model on Hugging Face:** [TextMiningStories/Mistral-Small-3.1-24B-goemotions](https://huggingface.co/TextMiningStories/Mistral-Small-3.1-24B-goemotions)
- **Training dataset:** [google-research-datasets/go_emotions](https://huggingface.co/datasets/go_emotions)
- **Base model:** [mistralai/Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)

---

## Citation

```bibtex
@inproceedings{demszky2020goemotions,
  title     = {GoEmotions: A Dataset of Fine-Grained Emotions},
  author    = {Demszky, Dorottya and Movshovitz-Attias, Dana and Ko, Jeongwook
               and Cowen, Alan and Nemade, Gaurav and Ravi, Sujith},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association
               for Computational Linguistics},
  year      = {2020},
  pages     = {4040--4054},
}
```
