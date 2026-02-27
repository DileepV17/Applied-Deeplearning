# Domain Adaptation of CLIP model using DANN (adversarial learning)
### Applied Deep Learning | WiSe 2026 | LMU Munich

**Project Members:** 
* Dileep Vemuri (12818965)
* Anjali Sarawgi (12690415)
* Duc-Anh Nguyen (12433139)


---

## 📌 Project Overview
This project evaluates strategies for adapting **CLIP (Contrastive Language-Image Pre-training)** to diverse visual domains. We explore the "domain gap" between real-world imagery and specialised distributions like **Infographic**, **Clipart**, and **Sketch**. Our methodology evolves from Zero-Shot baselines to Adversarial Adaptation (DANN) and a custom Diversity-Ensemble approach designed to capture robust features.

## 📊 Dataset: Domain Net
We use **Domain Net** (https://ai.bu.edu/M3SDA), a benchmark for multi-source domain adaptation.
* **Source Domain:** `Real`
* **Target Domains:** `Clipart`
* **Modality:** Image-Text (leveraging CLIP text-embeddings from class names).

---

## 🚀 Implementation Stages

### Stage 1: Zero-Shot Evaluation
Baseline performance of the frozen `ViT-B-32` model without any domain-specific training.

| Domain | Accuracy |
| :--- | :--- |
| **Real (Source)** | 78.63% |
| **Infograph (Target)** | 40.46% |
| **Clipart (Target)** | [TODO]% |
| **Clipart (Target)** | [TODO]% |
| **Clipart (Target)** | [TODO]% |
| **Clipart (Target)** | [TODO]% |
| **Clipart (Target)** | [TODO]% |

### Stage 2: Fine-Tuning & Ensemble Methods
We explore adaptation by freezing the CLIP backbone and training custom heads to mitigate catastrophic forgetting.
 
### Stage 3: Domain Adaptation (DANN)
To bridge the distribution gap, we implement a **Domain-Adversarial Neural Network**.
* **GRL:** A Gradient Reversal Layer is used to train a domain regressor, forcing the encoder to learn domain-invariant representations.
* **Alignment:** We use t-SNE to visualize how DANN aligns the feature clusters of the source and target domains.

---

## 🛠 Repository Structure
Following the mandatory course template:

```text
├── data/               # Dataset lists and loaders
├── models/             # DANN, MLP Heads, and Ensemble architectures
├── notebooks/          # t-SNE visualizations and EDA
├── scripts/            # Training scripts (DANN, Ensemble, Zero-shot)
├── utils/              # Gradient Reversal and Metrics logic
├── main.py             # Entry point for all experiments
├── requirements.txt    # Environment dependencies
└── README.md
```
---

⚙️ Setup and Execution
Environment:

```bash
pip install -r requirements.txt
```
Run Zero-Shot Baseline:

```bash

python scripts/zero_shot.py
```
Train DANN Adaptation:
```bash

python scripts/dannMain.py --target infograph
```
Train Ensemble with Diversity Loss:
```bash

python scripts/3subnetworks_wDL.py
```
