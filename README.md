
## Dataset

**UCI Heart Disease (Cleveland subset)**
- 303 patients, 13 clinical features
- Binary target: presence of heart disease (0 = no, 1 = yes)
- Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease)



## Setup

```bash
git clone https://github.com/YOUR_USERNAME/autogluon-exploration.git
cd autogluon-exploration
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/autogluon_heart_disease.ipynb
```

> AutoGluon is a large install (~6 GB). `best_quality` training takes ~5 min on a standard laptop.

---

## Results

| Preset | ROC-AUC | Accuracy | Training Time |
|--------|---------|----------|---------------|
| `best_quality` | **0.893** | **0.869** | ~4.5 min |
| `medium_quality` | 0.871 | 0.852 | ~1.2 min |
| `optimize_for_deployment` | 0.864 | 0.844 | ~0.8 min |
| Baseline (sklearn RF) | 0.857 | 0.836 | ~3 sec |

---

## Author

Ronald Mjonono, 473522
