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

### Stage 1: Zero-Shot Learning
Baseline performance of the pre-trained CLIP model. This highlights the inherent domain shift difficulty before any adaptation.

| Domain Type | Accuracy (Test Set) |
| :--- | :--- |
| **Real (Source)** | **0.78** |
| **Clipart** | 0.62 |
| **Painting** | 0.59 |
| **Sketch** | 0.55 |
| **Infograph** | 0.40 |

---

### Stage 2: Fine-Tuning Strategies

#### 2.1 Full CLIP Fine-Tuning
We unfreeze the entire CLIP visual encoder. Results show that **Label Smoothing (ls)** provides a marginal but consistent improvement across domains.

| Domain | lr | ls | Accuracy |
| :--- | :--- | :--- | :--- |
| **Real** | 0.001 | yes | **0.86** |
| **Clipart** | 0.001 | yes | **0.65** |
| **Infograph** | 0.001 | yes | **0.38** |
| **Real** | 0.001 | no | 0.85 |
| **Clipart** | 0.001 | no | 0.63 |
| **Infograph** | 0.001 | no | 0.37 |

#### 2.2 MLP Head Probing (Frozen Backbone)
We freeze the CLIP model and append an MLP head to mitigate catastrophic forgetting. **2-layer architectures** significantly outperformed 5-layer versions, which likely suffered from optimization difficulties on fixed features.


| Domain | lr | Layers | Scheduler | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Real** | 0.001 | 2 | StepLR | **0.84** |
| **Clipart** | 0.001 | 2 | StepLR | **0.59** |
| **Infograph** | 0.001 | 2 | StepLR | **0.36** |
| **Real** | 0.001 | 2 | Cosine | **0.84** |
| **Clipart** | 0.001 | 2 | Cosine | **0.59** |
| **Infograph** | 0.001 | 2 | Cosine | **0.36** |
| **Real** | 0.001 | 5 | StepLR | 0.73 |
| **Clipart** | 0.001 | 5 | StepLR | 0.43 |
| **Infograph** | 0.001 | 5 | StepLR | 0.23 |
| **Real** | 0.001 | 5 | Cosine | 0.73 |
| **Clipart** | 0.001 | 5 | Cosine | 0.42 |
| **Infograph** | 0.001 | 5 | Cosine | 0.20 |

#### 2.3 3-Model Ensemble (Frozen Backbone)
Using an ensemble of three parallel 2-layer MLP heads. 

| Domain | lr | Layers | Scheduler | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Real** | 0.001 | 2 | Cosine | **0.82** |
| **Clipart** | 0.001 | 2 | Cosine | **0.53** |
| **Infograph** | 0.001 | 2 | Cosine | **0.33** |

*(Note: We also introduced a Diversity Loss mechanism to force disagreement between the ensemble components, preventing representational collapse:)*
$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \cdot \max(0, \alpha - \text{std}(z_k))$$

---
### Stage 3: Domain Adaptation (DANN)
We implement adversarial learning to align source and target feature distributions using a **Gradient Reversal Layer (GRL)**. We experimented with both frozen and unfrozen CLIP backbones to evaluate the trade-off between feature adaptation and catastrophic forgetting.


#### 3.1 DANN with Frozen CLIP Encoder
The CLIP visual encoder is frozen, and we train MLP heads (configuration: 2 layers for feature extraction, 2 layers for classification, 3 layers for domain discrimination). We used an Adaptive Lambda scheduler (scaled to a maximum of 0.8) for the gradient reversal.

| Domain | Learning Rate | Layers | Lambda | Scheduler | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Real** | 0.01 | 2, 2, 3 | Adaptive | Cosine | 0.83 |
| **Clipart** | 0.01 | 2, 2, 3 | Adaptive | Cosine | 0.58 |
| **Real** | 0.001 | 2, 2, 3 | Adaptive | Cosine | **0.84** |
| **Clipart** | 0.001 | 2, 2, 3 | Adaptive | Cosine | **0.59** |

#### 3.2 DANN with Unfrozen CLIP Visual Encoder
In this two-stage fine-tuning approach, we unfreeze the CLIP encoder to allow deeper adaptation to the target domain distributions. We evaluated the impact of adding the adversarial domain loss against a baseline without it.

| Domain | Lambda | Domain Loss Added | Accuracy | Domain Gap |
| :--- | :--- | :--- | :--- | :--- |
| **Real** | Adaptive (max 0.3) | No | 0.84 | - |
| **Clipart** | Adaptive (max 0.3) | No | 0.58 | 26 pt. |
| **Real** | Adaptive (max 0.3) | Yes | 0.84 | - |
| **Clipart** | Adaptive (max 0.3) | Yes | 0.60 | 24 pt. |
| **Real** | Adaptive (max 0.6) | Yes | **0.83** | - |
| **Clipart** | Adaptive (max 0.6) | Yes | **0.62** | **21 pt. \*** |

**Key Finding:** Unfreezing the encoder and applying a higher Adaptive Lambda (max 0.6) successfully reduced the domain gap between Real and Clipart down to 21 percentage points, achieving the highest target accuracy (62%) for the Clipart domain in our experiments.

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
