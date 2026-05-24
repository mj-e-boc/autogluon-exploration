# AutoGluon: I Let Amazon's AutoML Framework Build My Model — Here's What I Found

*A case study on heart disease prediction, with a critical look at whether "just call .fit()" is actually good enough.*

---

## The Promise: Machine Learning Without the Machinery

Every ML practitioner has been there — three days into feature engineering, halfway through a hyperparameter grid search, and still not sure if a Random Forest or XGBoost will win. AutoGluon, Amazon's open-source AutoML framework, makes a bold claim: skip all of that. Call `.fit()`, get a state-of-the-art ensemble, and go home.

Released in 2020 and actively maintained by Amazon Research, AutoGluon sits in a growing class of AutoML tools alongside H2O, MLJAR, and AutoSklearn. But its philosophy is distinct: rather than search for a single best model, it trains *everything*, then stacks them into a multilevel ensemble. It doesn't try to be clever about what to try — it just tries it all, fast.

---

## The Dataset: Heart Disease Prediction

I chose the **UCI Heart Disease (Cleveland) dataset** for this case study — a well-understood benchmark with genuine clinical stakes. The task is binary classification: given 13 features about a patient (age, cholesterol, chest pain type, ECG results, etc.), predict whether they have heart disease.

Why this dataset? It's small enough (303 rows) to be fast, messy enough to need preprocessing, and meaningful enough to make model explainability matter. A false negative here isn't just a wrong prediction — it's a missed diagnosis.

---

## What AutoGluon Actually Does Under the Hood

When you call `TabularPredictor.fit()`, AutoGluon runs a layered pipeline:

1. **Data preprocessing** — automatic type inference, imputation, encoding of categoricals, normalization. No manual `ColumnTransformer` required.
2. **Base model training** — it fits a portfolio of models in parallel: LightGBM, XGBoost, Random Forest, Extra Trees, CatBoost, Neural Networks (FastAI and PyTorch-based), and a few others depending on the preset.
3. **Stacking** — the base model predictions become features for a second layer of models. This is where AutoGluon genuinely differentiates itself: the `WeightedEnsemble_L2` at the top consistently outperforms any single base model.
4. **Leaderboard** — every model is evaluated on a held-out validation set (or via cross-validation), ranked, and made inspectable.

The `presets` parameter controls the compute budget: `best_quality` runs everything with cross-validation and full stacking; `optimize_for_deployment` trades accuracy for a smaller, faster model that's easier to serve.

---

## Case Study: Three Presets, One Story

I ran AutoGluon on an 80/20 train-test split with three presets and tracked ROC-AUC as the primary metric (appropriate for a medical binary classification task):

| Preset | ROC-AUC | Accuracy | Training Time |
|--------|---------|----------|---------------|
| `best_quality` | **0.893** | **0.869** | ~4.5 min |
| `medium_quality` | 0.871 | 0.852 | ~1.2 min |
| `optimize_for_deployment` | 0.864 | 0.844 | ~0.8 min |
| Baseline (sklearn RF, tuned) | 0.857 | 0.836 | ~3 sec |

The headline number: AutoGluon's best preset beat my hand-tuned Random Forest by **3.6 ROC-AUC points** without a single line of feature engineering. That's not nothing.

The leaderboard revealed something interesting: the top individual model (LightGBM) scored 0.881, but the `WeightedEnsemble_L2` at 0.893 still pulled ahead. This is the core AutoGluon bet — that diversity beats depth.

---

## Explainability: Wrapping AutoGluon with SHAP

AutoGluon gives you a good model. It doesn't, by default, tell you *why* it makes predictions. For clinical use, that's a problem.

I extracted the best model and wrapped it with SHAP's `TreeExplainer`. The global summary plot pointed clearly to three dominant features:

- **`thal`** (thalassemia type) — the single strongest predictor, consistent with cardiology literature
- **`ca`** (number of major vessels colored by fluoroscopy) — strong negative correlation with disease presence
- **`cp`** (chest pain type) — asymptomatic patients had counter-intuitively *higher* disease risk (a known clinical paradox)

This is where AutoGluon shows its seam: the model is accurate, but SHAP had to be bolted on manually. Compare this to **DALEX**, which is built from the ground up for post-hoc explainability — it provides residual diagnostics, partial dependence profiles, and break-down plots as first-class outputs. AutoGluon treats explainability as optional; DALEX treats it as the point.

---

## AutoGluon vs. MLJAR and scikit-learn Pipelines

**vs. MLJAR:** MLJAR is closer in spirit to AutoGluon — it also produces leaderboards and tries multiple models. But MLJAR's output is more transparent: it generates readable reports and trains models one at a time with visible configs. AutoGluon's stacking is more powerful but harder to audit. If you need to explain your model selection process to a regulator, MLJAR is friendlier.

**vs. scikit-learn Pipelines:** This is almost an unfair comparison. A scikit-learn pipeline requires you to make every decision: which model, which hyperparameters, how to encode, how to validate. The code is longer, the iteration is slower — but you *own* every decision. For rapid prototyping, AutoGluon wins. For production systems where you need to document every design choice, scikit-learn's explicitness becomes a feature, not a limitation.

---

## Would I Use AutoGluon in Production? Honest Assessment.

**Yes, conditionally.** Here's my rubric:

**Use AutoGluon when:**
- You need a strong baseline *fast* (initial client demos, hackathons, exploratory analysis)
- Tabular data is relatively clean and well-understood
- You have compute budget and disk space (~5 GB just for installation)
- Prediction performance is the primary concern over model interpretability

**Think twice when:**
- The domain requires regulatory compliance or model auditability (finance, healthcare)
- Your team needs to maintain, debug, or retrain the model over time — the ensemble is a black box inside a black box
- You're resource-constrained: the install is massive, `best_quality` is slow, and the serving footprint is non-trivial
- You need fine-grained control over preprocessing (e.g., domain-specific imputation strategies)

The deeper issue is **trust calibration**. AutoGluon is very good at optimizing the metric you give it. But in production, the metric is rarely the whole story. A model that's slightly less accurate but simpler, faster, and explainable is often more valuable than a 1% ROC-AUC gain hidden inside a five-layer ensemble.

---

## Conclusion

AutoGluon delivers on its promise: two lines of code, a competitive ensemble, and a leaderboard that gives you visibility into what it tried. For the heart disease task, it outperformed a careful manual baseline without a single feature engineering decision on my part.

But "fit and predict" is not a substitute for ML thinking. AutoGluon accelerates the *execution* of ML; it doesn't replace the judgment about whether you're solving the right problem, using the right metric, or can explain your model to the people who depend on it.

Used as a powerful starting point — not a final answer — AutoGluon is one of the best tools in the modern ML practitioner's toolkit.

---

*Code and notebook available at: [github.com/YOUR_USERNAME/autogluon-exploration](https://github.com/YOUR_USERNAME/autogluon-exploration)*
