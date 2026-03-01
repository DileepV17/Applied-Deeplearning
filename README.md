# Domain Adaptation of CLIP model using DANN (adversarial learning)
### Applied Deep Learning | WiSe 2026 | LMU Munich

**Project Members:** 
* Dileep Vemuri (12818965)
* Anjali Sarawgi (12690415)
* Duc-Anh Nguyen (12433139)


---

## 📌 Project Overview
This project evaluates strategies for adapting **CLIP (Contrastive Language-Image Pre-training)** to diverse visual domains. We explore the "domain gap" between real-world imagery and specialized distributions like infographics, clipart, and sketches using the **DomainNet** (https://ai.bu.edu/M3SDA) benchmark.

---

## 1. Model Architecture
Our project relies on the pre-trained **CLIP (ViT-B/32)** model as the foundation.  To prevent catastrophic forgetting, the visual encoder is frozen during adaptation. We explored three primary architectural extensions:

1. **Linear Probing / MLP Heads:** A 2-layer Multi-Layer Perceptron (MLP) trained exclusively on the source domain.
2. **Domain-Adversarial Neural Network (DANN):**  We attach a domain classifier network via a **Gradient Reversal Layer (GRL)**. The GRL reverses gradients during backpropagation, forcing the feature extractor to learn representations that are discriminative for the main task but invariant across the source and target domains.
3. **Multi-Subnetwork Ensemble:** An ensemble of $K=3$ parallel subnetworks. Input images are augmented into two distinct views, passed through the frozen backbone, and processed by the subnetworks. 
---

## 2. Training and Validation Curves

### DANN Adaptation Curves
![DANN Loss Curves](data_statistics/totalloss.png)
*Figure 1: Total loss for the DANN classifier. The adversarial domain loss stabilizes as the feature extractor successfully learns domain-invariant representations.*
![DANN Loss Curves](data_statistics/domainaccuracy.png)
*Figure 2: Domian Accuracy initially starts high and oscillates around 0.5 indicating domain confusion is working correctly.*
![DANN Loss Curves](data_statistics/valaccuracies.png.png)
*Figure 2: Source(Real_images) and Target(Clipart_images) validation accuracy curves.*

---
## 3. Ablation Studies
We conducted ablation studies to isolate the impact of our adaptation techniques. The table below compares the performance of the frozen CLIP model with and without our specific domain adaptation strategies on the `Infograph` target domain.

| Method | Source Acc (Real) | Target Acc (Clipart) | 
| :--- | :--- | :--- | :--- |
| **Zero-Shot (Baseline)** | 78.63% | 62.00% | 
| **Source Fine-Tuning (Unfreeze only visual encoder No Adapt)** | 86.00% | 65.00% |
| **Source Fine-Tuning with 2 MLP (No Adapt)** | 84.00% | 59.00% |
| **Source Fine-Tuning with 3 ensembles (each 2 MLP (No Adapt))** | 82.00% | 53.00% | 
| **DANN (Adversarial Adapt)** | 83.00% | 62.00% | 



---

## 4. Discussion of Results and Limitations

### Results Analysis
* **Feature Alignment:** t-SNE visualizations  confirm that DANN effectively clusters target domain features closer to the source domain features compared to the zero-shot baseline.


### Limitations & Future Work
1. **Catastrophic Forgetting:** Early experiments with an unfrozen CLIP encoder resulted in severe degradation of pre-trained knowledge. While freezing the backbone solved this, it limited the theoretical maximum adaptation capacity of the model.
2. **Modal Imbalance:** Currently, text embeddings are generated solely from class names. The project guidelines suggested using paired captions (via BLIP/COCO). Integrating rich captions could further improve alignment but was computationally prohibitive within our current setup.
3. **Batch Size Constraints:** DANN relies on large batch sizes to accurately estimate domain distributions. Due to hardware limitations, our batch size was restricted to 64, which may have introduced noise into the adversarial training process.

---

## 🛠 Repository Structure
Following the mandatory course template:

```text
├── data/               # Dataset lists and loaders
├── models/             # DANN, MLP Heads, and Ensemble architectures
├── notebooks/          # t-SNE visualizations and EDA
├── scripts/            # Training scripts (DANN, Ensemble, Zero-shot)
├── utils/              # Helper functions: Gradient Reversal and Metrics logic
├── configs/            # YAML/JSON configs files
├── requirements.txt    # Environment dependencies
├── README.md
└── main.py             # Entry point for all experiments

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
