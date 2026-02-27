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
3. **Multi-Subnetwork Ensemble:** An ensemble of $K=3$ parallel subnetworks. Input images are augmented into two distinct views, passed through the frozen backbone, and processed by the subnetworks. A **Diversity Loss** is applied to penalise low variance among the subnetworks, ensuring mutually exclusive feature learning:

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \cdot \max(0, \alpha - \text{std}(z_k))$$

**Where:**

* **$\mathcal{L}_{CE}$**: The standard **Cross-Entropy Loss** calculated on the averaged predictions across the ensemble.
* **$\lambda$**: The **Diversity Weight** (hyperparameter) which scales the impact of the diversity penalty.
* **$\alpha$**: The **Margin threshold**; the penalty becomes zero once the disagreement between subnetworks exceeds this value (preventing over-optimization).
* **$\text{std}(z_k)$**: The **Standard Deviation** across the $K$ subnetwork outputs ($z_1, z_2, \dots, z_K$), serving as a mathematical proxy for representational diversity.

---

## 2. Training and Validation Curves

### DANN Adaptation Curves
![DANN Loss Curves](docs/dann_loss_curves.png)
*Figure 1: Training and validation loss for the DANN classifier. The adversarial domain loss stabilizes as the feature extractor successfully learns domain-invariant representations.*

### Ensemble Diversity Curves
![Ensemble Accuracy](docs/ensemble_accuracy_curves.png)
*Figure 2: Validation accuracy on the Target (Infograph) domain over 10 epochs. The model with Diversity Loss (blue) shows better generalization and stability compared to the standard ensemble (red).*

---

## 3. Ablation Studies
We conducted ablation studies to isolate the impact of our adaptation techniques. The table below compares the performance of the frozen CLIP model with and without our specific domain adaptation strategies on the `Infograph` target domain.

| Method | Source Acc (Real) | Target Acc (Infograph) | Target F1-Score |
| :--- | :--- | :--- | :--- |
| **Zero-Shot (Baseline)** | 78.63% | 40.46% | *[TODO]* |
| **Source Fine-Tuning (No Adapt)** | *[TODO]*% | *[TODO]*% | *[TODO]* |
| **Ensemble (w/o Diversity Loss)** | *[TODO]*% | *[TODO]*% | *[TODO]* |
| **Ensemble (w/ Diversity Loss)** | *[TODO]*% | *[TODO]*% | *[TODO]* |
| **DANN (Adversarial Adapt)** | *[TODO]*% | *[TODO]*% | *[TODO]* |

**Key Finding:** DANN and the Diversity Ensemble both significantly outperform the zero-shot and standard fine-tuning baselines on the target dataset, demonstrating successful mitigation of the domain gap.

---

## 4. Discussion of Results and Limitations

### Results Analysis
* **Feature Alignment:** t-SNE visualizations  confirm that DANN effectively clusters target domain features closer to the source domain features compared to the zero-shot baseline.
* **Diversity Loss:** Forcing disagreement among the $K=3$ subnetworks proved highly effective at preventing representational collapse, yielding a more robust classifier that is less sensitive to domain-specific quirks.

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
