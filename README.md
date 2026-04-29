# 🎤 VR Speech Anxiety Analysis
Physiological Biomarker Validation for Wearable-Based Anxiety Screening

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ZariaZhao/VR-Anxiety-Analysis/blob/main/speech_anxiety_vr_analysis.ipynb)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Exploratory%20Research-purple)]()

[English](#-vr-speech-anxiety-analysis) | [中文说明](#-项目简介中文)

---

> Validating the technical feasibility of wearable + VR as a low-cost,
> objective speech anxiety screening and personalised intervention matching system.

---

## 🌟 Highlights

- 📡 **Objective Biomarker Identified**: Heart rate transition (B→C) is the
  strongest anxiety predictor (26.3% feature importance) — dynamic HR
  adaptability outperforms static measurements
- 🔬 **Subjective-Objective Dissociation**: 30% of participants show normal
  self-report but elevated physiological response — questionnaires alone
  miss a meaningful proportion of anxious individuals
- 🤖 **Predictive Model**: Random Forest achieves RMSE=0.244 using
  physiological + personality features (heart rate signals = 63.9% of
  predictive power)
- 👥 **Three Anxiety Phenotypes**: GMM clustering identifies distinct
  response profiles, each peaking in a different VR scenario —
  supporting personalised scenario matching
- ✅ **Manipulation Check Validated**: VR scenarios confirmed as
  emotionally distinct (Pleasure F=7.76 p<0.001; Arousal F=8.07 p<0.001)

---

## 🎯 Project Overview

### The Problem

Public speaking anxiety affects 63.9% of university students, yet existing
interventions face two fundamental limitations:

**Measurement gap**: Over-reliance on self-report questionnaires misses
subclinical anxiety — 30% of individuals show normal self-reports while
physiological signals indicate stress activation.

**One-size-fits-all protocols**: Generic exposure therapy fails to account
for individual differences in anxiety response patterns, leading to
inconsistent outcomes.

### The Approach

This project validates a wearable + VR pipeline as a low-cost alternative:
Participant wears HR monitor
↓
90-second VR scenario B (depressing) → transition to scenario C (tense)
↓
Real-time heart rate trajectory captured
↓
Random Forest outputs anxiety risk score
↓
GMM assigns phenotype → matched to optimal training scenario

No questionnaire required for initial screening.

### Research Questions

1. Can physiological signals (HR, speech rate, voice stability) predict
   subjective anxiety levels without relying on self-report?
2. Do distinct anxiety response phenotypes exist, and do they respond
   differently to different VR emotional contexts?
3. Is there evidence of subjective-objective dissociation that justifies
   objective physiological monitoring?

---

## 🔑 Key Findings

| Finding | Method | Result |
|---------|--------|--------|
| **Scenario manipulation valid** | Repeated measures ANOVA | Pleasure η²=0.290, Arousal η²=0.298, both p<0.001 |
| **Scenarios affect internal experience** | Repeated measures ANOVA | Subjective performance η²=0.050 vs objective η²=0.003 |
| **Pleasure drives self-perceived performance** | Pearson correlation | r²=0.217, p<0.05 — strongest scenario predictor |
| **HR transition = strongest biomarker** | Random Forest (feature importance) | HeartRate_diff_B→C = 26.3%, HeartRateB = 19.8% |
| **HR signals dominate prediction** | Random Forest (category importance) | Heart rate = 63.9% of model predictive power |
| **Anxiety prediction feasible** | Random Forest regression (5-fold CV) | RMSE = 0.244 ± 0.084 |
| **Neuroticism links to anxiety > performance** | Pearson correlation | Anxiety r=+0.326 vs performance r=-0.093 |
| **30% subjective-objective dissociation** | Bland-Altman analysis | Questionnaire misses physiologically anxious individuals |
| **Three anxiety phenotypes identified** | GMM clustering (k=3) | Comfort-Dependent 15% / Pressure-Activated 45% / Excitement-Driven 40% |
| **Phenotypes peak in different scenarios** | Performance trajectory analysis | Each phenotype has a distinct optimal VR scenario |

---

## 📊 Dataset

**Design**: 4×20 repeated measures within-subject study

- **Participants**: 20 university students (ethics-approved)
- **VR Scenarios** (Russell's Circumplex Model — Pleasure × Arousal):

| Scenario | Emotional Profile | Role |
|----------|------------------|------|
| A — Cozy | High pleasure × Low arousal | Baseline comfort |
| B — Depressing | Low pleasure × Low arousal | Primary stressor — critical for HR prediction |
| C — Tense | Low pleasure × High arousal | Pressure escalation |
| D — Exciting | High pleasure × High arousal | Positive activation control |

- **Total observations**: 80 (20 participants × 4 scenarios)
- **Features**: 49 dimensions across 5 categories

| Category | Features | Examples |
|----------|----------|---------|
| Personality | 5 | Big Five traits (Neuroticism, Extraversion...) |
| Physiology | 16 | Heart rate × 4 scenarios + transition differences |
| Acoustics | 12 | Speech rate, voice stability (jitter, shimmer) |
| Anxiety scales | 8 | Subjective (questionnaire) + Objective (physiological composite) |
| Performance | 8 | Self-rated confidence + evaluator-rated quality |

---

## 🔬 Methodology

### Analysis Pipeline

Manipulation Check
└── Verify VR scenarios perceived as designed (Pleasure × Arousal ANOVA)
Scenario Effects (Repeated Measures ANOVA)
└── Subjective vs objective performance × scenario type
└── Key finding: subjective-objective dissociation
Bivariate Analysis (Section 5.5)
├── Immersion → subjective performance (r=0.224, p=0.046)
├── Scenario features → performance (Pleasure r²=0.217)
├── Physiological signals → anxiety/performance
├── Personality → anxiety & performance
└── Master correlation heatmap → feature selection basis
Random Forest Regression
└── Target: Subjective_Anxiety
└── Features: 20 (HR × 4 scenarios + transitions, SpeechRate × 4,
VoiceStability × 4, Big Five × 3)
└── Validation: 5-fold cross-validation
GMM Clustering
└── Features: Neuroticism, HeartRate_diff_A_B, VoiceStabilityA
└── k=3 (BIC-optimal)
└── Labels assigned from centroid characteristics + Section 6 results
Phenotype Validation
└── Performance trajectory by phenotype × scenario
└── Confirms distinct scenario preferences per phenotype


### Random Forest: Feature Importance

| Rank | Feature | Importance | Signal Type |
|------|---------|-----------|-------------|
| 1 | HeartRate_diff_B_C | 26.3% | HR transition (B→C) |
| 2 | HeartRateB | 19.8% | HR in depressing scenario |
| 3 | SpeechRateC | 8.7% | Speech rate under tension |
| 4 | VoiceStabilityD | 7.5% | Vocal stability (exciting) |

**By signal category**:
- Heart rate (scenarios + transitions): 63.9%
- Speech rate: 18.4%
- Voice stability: 12.3%
- Personality traits: 5.9%

**Model performance**:
- CV RMSE: 0.244 ± 0.084 (5-fold, primary metric)
- Note: CV R² negative due to n=20 fold size (~4 per fold) —
  feature importance from full-data fit is the more reliable metric

### GMM Phenotypes

| Phenotype | n | Neuroticism | HR Change A→B | Optimal Scenario |
|-----------|---|------------|--------------|-----------------|
| Comfort-Dependent | 3 (15%) | Highest (+0.86) | Large negative (HR rose sharply) | A — Cozy |
| Pressure-Activated | 9 (45%) | Moderate (+0.49) | Positive (HR stable) | C — Tense |
| Excitement-Driven | 8 (40%) | Lowest (-0.87) | Moderate negative | D — Exciting |

*Labels are researcher-assigned based on cluster centroid characteristics
and performance trajectories. Exploratory only — n per group is very small.*

---

## 📈 Visual Outputs

| Figure | Description |
|--------|-------------|
| `manipulation_check.png` | Pleasure/Arousal ratings by scenario + Russell Circumplex plot |
| `performance_comparison.png` | Subjective vs objective performance across 4 scenarios |
| `physiological_indicators.png` | HR, speech rate, voice stability trajectories |
| `anxiety_dissociation.png` | Bland-Altman agreement analysis |
| `scenario_features_performance.png` | Pleasure/Arousal/Immersion → performance |
| `physio_anxiety_performance.png` | Physiological signals → anxiety & performance |
| `personality_anxiety_performance.png` | Big Five × 3 outcomes heatmap |
| `master_correlation_heatmap.png` | All predictors × all outcomes + feature ranking |
| `random_forest_anxiety.png` | Feature importance bar chart + category pie chart |
| `phenotype_discovery.png` | GMM distribution + cluster profiles |
| `heartrateB_correlation.png` | HeartRateB biomarker scatter plot |
| `phenotype_validation.png` | Performance trajectory by phenotype × scenario |

---

## ⚠️ Limitations

- **n=20**: All findings are exploratory — insufficient power for
  definitive statistical conclusions
- **Single anxiety composite**: Subjective_Anxiety is one overall score,
  not scenario-specific; limits granularity of prediction
- **Objective_Anxiety**: Derived from physiological composite score,
  not independently validated against gold-standard measures
- **Small phenotype groups**: Comfort-Dependent n=3 — treat with caution
- **No intervention data**: Cannot confirm that phenotype-matched training
  improves outcomes — requires randomised controlled trial

---

## 🚀 Quick Start

### Option A: Google Colab (Recommended)

1. Click the Colab badge above
2. Upload `data/001.xlsx` via the left sidebar
3. `Runtime` → `Run all` (~5 minutes)

### Option B: Run Locally

```bash
git clone https://github.com/ZariaZhao/VR-Anxiety-Analysis.git
cd VR-Anxiety-Analysis
pip install -r requirements.txt
jupyter notebook speech_anxiety_vr_analysis.ipynb
```

### Option C: Quick ML Demo

```bash
python src/simple_prediction_demo.py
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| Data processing | pandas, numpy |
| Machine learning | scikit-learn (Random Forest, GMM) |
| Statistics | scipy, pingouin (rm-ANOVA, Bland-Altman) |
| Visualisation | matplotlib, seaborn |
| Environment | Jupyter Notebook, Google Colab |

---

## 📁 Repository Structure
VR-Anxiety-Analysis/
├── speech_anxiety_vr_analysis.ipynb  # Main analysis notebook
├── src/
│   └── simple_prediction_demo.py     # Standalone ML demo
├── outputs/                          # All generated figures (14 files)
├── data/
│   └── 001.xlsx                      # Research data (not public)
└── README.md