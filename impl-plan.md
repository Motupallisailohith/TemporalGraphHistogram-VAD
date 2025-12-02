# HistoGraph Implementation Plan
## Temporal Graph Networks for Dynamic Histogram Evolution in Video Anomaly Detection

---

## Executive Summary

This implementation plan transforms the HistoGraph project proposal into an executable roadmap, addressing critical gaps and red flags in the original design. The plan emphasizes rigorous dataset handling, multi-level evaluation metrics, comparative baselines, and architectural depth necessary for achieving the proposed 5–15% AUC improvement over state-of-the-art methods.

---

## 1. Datasets Strategy

### 1.1 Primary Datasets (Evaluation)

#### UCSD Pedestrian Dataset (Ped2)
- **Characteristics:** 98 normal training clips, 32 test clips (mixed normal/anomalous)
- **Resolution:** 360×240 pixels, 10 fps, 198 frames per clip
- **Anomaly Types:** Loitering, running, sudden crowd appearance
- **Usage:** Primary benchmark for frame-level AUC computation; baseline comparison against Shi et al. (2023)
- **Implementation Notes:**
  - Frame extraction via OpenCV; normalize to [0,1] for all feature channels
  - Temporal window size: 32 frames (3.2 seconds at 10 fps)
  - Train on 70% of normal frames; validate on 30%; test on all anomalous frames
  - Metric: Frame-level AUC (ROC-AUC with pixel-wise anomaly scores)

#### ShanghaiTech Campus Dataset
- **Characteristics:** 330 normal training videos, 107 test videos (anomalies: fighting, theft, running)
- **Resolution:** Variable (300–720 pixels), 25 fps, 1–10 min duration
- **Rationale:** Tests robustness across diverse scenes, lighting conditions, camera motion
- **Usage:** Secondary validation; assess generalization beyond pedestrian-centric anomalies
- **Implementation Notes:**
  - Downsample videos to consistent 480×360 resolution to reduce memory overhead
  - Temporal window: 50 frames (2 seconds at 25 fps)
  - Frame-level AUC and video-level AUC comparison

#### Avenue Dataset
- **Characteristics:** 16 training videos, 21 test videos (anomalies: loitering, running, illegal parking)
- **Resolution:** 640×480 pixels, 30 fps
- **Rationale:** SOTA comparison benchmark per proposal; validate scalability and consistency
- **Usage:** Final SOTA validation and qualitative visualization
- **Implementation Notes:**
  - Temporal window: 60 frames (2 seconds at 30 fps)
  - Cross-dataset evaluation metric: zero-shot transfer feasibility

### 1.2 Auxiliary Datasets (Qualitative Analysis)

#### AVA (Atomic Visual Actions) Dataset
- **Usage:** Generate qualitative anomaly heatmaps under lighting/scene shifts (non-standard for anomaly detection)
- **Implementation:** Select 50–100 clips with high illumination variance; overlay HistoGraph anomaly scores as heatmaps
- **Rationale:** Demonstrates scene-level understanding beyond pixel anomalies

#### UCF-Crime Dataset
- **Usage:** Optional ablation; compare against action-centric methods
- **Note:** Weak labels only; use for conceptual validation, not primary evaluation

### 1.3 Dataset Versioning and Splits

```
UCSD Ped2:
├── Train (Normal): frames 0–3000 (70% of 4285 normal frames)
├── Val (Normal): frames 3001–4285 (30% of normal)
└── Test (Mixed): all 32 anomalous clips

ShanghaiTech:
├── Train (Normal): 330 videos (per dataset standard)
├── Val (Normal): 20% sample from training
└── Test (Mixed): 107 videos

Avenue:
├── Train (Normal): 16 videos
├── Val (Normal): sampled frames from training
└── Test (Mixed): 21 videos
```

**Reproducibility:** Store dataset indices in `data/splits/` as JSON; log random seeds for all train/val/test splits.

---

## 2. Applications and Deployment Scenarios

### 2.1 Primary Applications

**Real-Time Surveillance Monitoring**
- **Scenario:** Urban intersection, airport terminal, or campus security
- **Requirements:** Frame-level latency ≤100 ms (CPU-optimized per proposal)
- **Deployment:** Standalone inference on edge devices (Jetson Nano, x86 embedded systems)
- **Metrics:** FPS, memory footprint (< 2 GB), power consumption

**Post-Event Investigation**
- **Scenario:** Anomaly heatmap generation for security review (qualitative)
- **Requirements:** Batch processing of recorded footage; detailed per-frame confidence scores
- **Output:** Visual overlays + CSV logs with timestamps and anomaly scores
- **Metrics:** Generation speed, heatmap interpretability, false-positive filtering

**Multi-Camera Coordination**
- **Scenario:** Correlate anomalies across multiple camera streams
- **Requirements:** Graph-based scene similarity; cross-camera anomaly propagation
- **Advanced Feature:** Temporal alignment across cameras (future work)

### 2.2 Operational Metrics (Beyond AUC)

| Metric | Target | Justification |
|--------|--------|---------------|
| Inference latency | ≤100 ms/frame (CPU) | Real-time deployment requirement |
| Memory footprint | ≤500 MB | Edge device constraints |
| Model size | ≤50 MB | Embedded deployment; faster loading |
| Precision (at 90% recall) | ≥75% | Reduces false alarms in operations |
| Video-level AUC | ≥90% | Segment-based decision making |

---

## 3. Improvement Ideology and Architectural Depth

### 3.1 Core Innovation: Why Histograms Over Objects

**Conceptual Progression:**
1. **Pixel-Level (Shi et al. 2023):** Dense 4D tensors; compute-heavy; misses global patterns
2. **Object-Level (Arnab et al. 2021):** Sparse but detection-dependent; O(N²) edges; fails under occlusion
3. **Histogram-Level (HistoGraph):** Fixed-dimensional scene representation; robust to detection failure; O(1) node complexity

**Why EMD-Weighted Edges Matter:**
- EMD (Earth Mover's Distance) captures distributional shifts gracefully
- Sinkhorn approximation: O(d³ log d) via entropic regularization vs. O(d⁴) for standard EMD
- Differentiable: enables backpropagation through edge weights

### 3.2 Architectural Depth: Multi-Scale Feature Fusion

```
Layer 1 (Feature Extraction):
├── Color: HSV (32 bins) + RGB (32 bins) → 96-dim vector
├── Motion: Farneback optical flow (magnitude histogram) → 32 bins
├── Texture: Local Binary Patterns (8 neighbors) → 59 bins
└── Aggregation: Concatenate → 187-dim feature vector g_t

Layer 2 (Graph Construction):
├── Nodes: 187 histogram bins (fixed across time)
├── Edges: EMD(g_t, g_{t-1}) + Sinkhorn weighting
├── Temporal Connectivity: Full connectivity across T-frame window
└── Storage: PyTorch Geometric Data objects with edge weights

Layer 3 (Message Passing):
├── Input: Node features (bin values) + edge weights (EMD)
├── GNN Layer 1: MLP embedding (187 → 64 dims)
├── GNN Layer 2: Graph convolution (GAT or GraphSAGE)
├── GRU Memory: Maintain hidden state h_t across sequences
└── Output: Reconstructed bin values + reconstruction error

Layer 4 (Anomaly Scoring):
├── Primary: L2 reconstruction error per frame
├── Auxiliary: Contrastive loss (normal vs. pseudo-anomalies)
├── Aggregation: Smooth anomaly score with temporal median filtering
└── Thresholding: Otsu's method or ROC-curve optimization
```

### 3.3 Stretch Goals for Depth

**Tier 1 (Core):** Sinkhorn EMD acceleration
- Entropic regularization parameter: λ ∈ [0.01, 0.1]
- Sinkhorn iterations: 100–200 (trade-off between speed and accuracy)

**Tier 2 (Extended):** Frame gating by optical flow
- Skip frames with optical flow magnitude < threshold (e.g., 0.05 pixels/frame)
- Expected speedup: 20–40% on static scenes
- Risk: May miss slow anomalies; requires careful threshold tuning

**Tier 3 (Advanced):** Hierarchical histograms
- Multi-resolution HSV (8, 16, 32, 64 bins in parallel)
- Learnable bin importance weighting
- Trade-off: +2–3% AUC vs. 1.5× memory overhead

---

## 4. Comparability Framework

### 4.1 Baseline Models (Required for Fair Comparison)

| Baseline | Reference | Rationale | Expected AUC |
|----------|-----------|-----------|--------------|
| **Simple Histogram Autoencoder** | Proposed | Ablation for graph necessity | 72–76% |
| **Multi-Pretext Learning** | Shi et al. (2023) | SOTA pixel-level | 78–86% |
| **Object-Graph TGN** | Arnab et al. (2021) | SOTA graph-level | 80–85% |
| **Optical Flow + GMM** | Classical | Robustness benchmark | 75–82% |
| **Variational Autoencoder (VAE)** | Standard | Unsupervised baseline | 73–78% |
| **HistoGraph (Proposed)** | This work | Target | 83–95% (5–15% gain) |

### 4.2 Fair Comparison Criteria

**Data Handling:**
- Identical train/val/test splits for all methods
- Same preprocessing (normalization, frame rate)
- No cherry-picked hyperparameters for competitors

**Evaluation Protocol:**
- Frame-level AUC (primary metric)
- Video-level AUC (secondary)
- Per-anomaly-type breakdown (e.g., AUC for running vs. loitering)
- ROC curve plotting with 95% confidence intervals

**Computational Fairness:**
- All models trained on same hardware (GPU/CPU)
- Training iterations synchronized (e.g., 100k iterations vs. 100 epochs—whichever is larger)
- Hyperparameter tuning budget: 200 trials via Optuna (shared across all methods)

### 4.3 Significance Testing

- Perform McNemar's test on frame-level predictions (α = 0.05)
- Report 95% confidence intervals via bootstrapping (1000 resamples)
- Document if improvement is statistically significant

---

## 5. Evaluation Metrics (Comprehensive)

### 5.1 Primary Metrics

**Frame-Level AUC (ROC-AUC)**
- Definition: Area under the Receiver Operating Characteristic curve for anomaly score thresholding
- Computation: Scikit-learn `roc_auc_score()` with frame-wise labels
- Rationale: Standard in literature; threshold-independent
- Target: Achieve 83–95% across UCSD Ped2, ShanghaiTech, Avenue

**Video-Level AUC**
- Definition: AUC computed on aggregated video-level anomaly scores (max or mean per video)
- Rationale: Practical for surveillance systems making per-video decisions
- Expected: Typically 5–10% higher than frame-level AUC

### 5.2 Secondary Metrics

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Precision @ 90% Recall** | TP / (TP + FP) at recall = 0.9 | Reduces false alarms in deployment |
| **Average Precision (AP)** | Area under P-R curve | Balances precision and recall |
| **AUROC (95% CI)** | Via bootstrapping (1000 resamples) | Confidence interval for robustness |
| **Per-Anomaly-Type AUC** | Stratified by anomaly class | Granular performance understanding |

### 5.3 Computational Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Inference latency | ≤100 ms/frame | `time.perf_counter()` on CPU (Intel i7 equiv.) |
| Training time | ≤4 hours (UCSD Ped2) | Logged per epoch in W&B |
| Memory (train) | ≤8 GB | `tracemalloc` or GPU memory monitoring |
| Memory (inference) | ≤500 MB | Peak memory during forward pass |
| Model size | ≤50 MB | Checkpoint file size on disk |

### 5.4 Qualitative Metrics

**Heatmap Quality (Visual Assessment)**
- Overlay anomaly scores on video frames
- Qualitative assessment: Do heatmaps align with ground-truth anomalies?
- Compute spatial overlap with ground-truth masks (if available) using Intersection-over-Union (IoU)

**Cross-Dataset Generalization**
- Train on UCSD Ped2 → Test on ShanghaiTech (zero-shot)
- Measure performance drop as a % of in-distribution AUC
- Target: <15% performance drop for robust models

---

## 6. Red Flags in Original Proposal and Mitigation Strategies

### 🚩 Red Flag 1: Vague Improvement Target ("Aims to achieve 5–15%")

**Issue:** The proposal states "aims to achieve" rather than committing to specific, measured targets. This undermines rigor and makes success evaluation ambiguous.

**Mitigation:**
- Define tier-based success criteria:
  - **Minimum:** 2% improvement over Shi et al. (83% AUC on UCSD Ped2)
  - **Target:** 5% improvement (83% → 88% AUC)
  - **Stretch:** 10% improvement (83% → 93% AUC)
- Log all intermediate results in W&B with date/time stamps
- Perform statistical significance testing (McNemar's test, p < 0.05)

### 🚩 Red Flag 2: Unclear Baseline Model Definition

**Issue:** No clear specification of what "state-of-the-art" baseline means. Shi et al. (2023) multi-pretext learning may not be reproducible or comparable.

**Mitigation:**
- Implement 5 baselines (see Section 4.1):
  - Simple histogram autoencoder (own implementation)
  - Re-implement or use published code for Shi et al. (2023)
  - Use Arnab et al. (2021) object-graph code if available (GitHub search)
  - Fallback: Compare against classical optical flow + Gaussian Mixture Models
- Document all baseline implementations in `baselines/` folder with reproducibility notes
- Ensure identical dataset splits for fair comparison

### 🚩 Red Flag 3: Insufficient Feature Engineering Justification

**Issue:** The proposal lists HSV, RGB, LBP, and optical flow but doesn't justify why each is necessary. Could be over-engineering.

**Mitigation:**
- **Ablation Study (Mandatory):**
  - Model A: HSV only → AUC on UCSD Ped2
  - Model B: HSV + RGB → AUC
  - Model C: HSV + RGB + Optical Flow → AUC
  - Model D: HSV + RGB + Optical Flow + LBP → AUC
  - Analyze incremental gains; drop features with <1% improvement-to-complexity ratio
- Document ablation results in a table within the final report

### 🚩 Red Flag 4: Sinkhorn EMD is "Stretch Goal," Not Core

**Issue:** The proposal relegates EMD acceleration to a "stretch goal," but EMD is central to the graph construction. Without optimization, training may be prohibitively slow.

**Mitigation:**
- **Move Sinkhorn EMD to Milestone 2 (Core):**
  - Implement both standard EMD (reference) and Sinkhorn EMD (optimized)
  - Benchmark: Measure wall-clock time for 1000 EMD computations on 187-dim histograms
  - Target: Sinkhorn EMD ≤5 ms per edge weight (vs. standard EMD potentially ≤50 ms)
- **Contingency:** If Sinkhorn is too slow, fall back to Sliced Wasserstein Distance (1D projections, much faster)

### 🚩 Red Flag 5: Frame Gating Risks Missing Slow Anomalies

**Issue:** Optical flow-based frame skipping could miss slow, anomalous behavioral changes (e.g., person gradually lying down).

**Mitigation:**
- Do NOT skip frames blindly
- Instead, implement adaptive gating:
  - Compute optical flow magnitude per frame
  - Skip only if magnitude < threshold (e.g., 0.05) AND previous 3 frames also skipped
  - Fall back to full frame processing if confidence drops below 0.6
- Ablation: Compare full vs. adaptive gating on Avenue dataset (which has slow anomalies)
- If gating causes >2% AUC drop, disable it

### 🚩 Red Flag 6: No Cross-Dataset Generalization Strategy

**Issue:** Training on UCSD Ped2 and testing on ShanghaiTech/Avenue is the standard, but no discussion of domain adaptation or transfer learning.

**Mitigation:**
- **Experiment 1:** Train on UCSD Ped2 alone → Test on ShanghaiTech (measure transfer performance drop)
- **Experiment 2:** If drop >15%, apply simple domain adaptation:
  - Fine-tune model on 10% of ShanghaiTech normal frames for 1 epoch
  - Measure AUC recovery
- **Experiment 3:** Multi-source training:
  - Train jointly on UCSD Ped2 + ShanghaiTech normal frames
  - Test on Avenue (true zero-shot)
  - Report AUC and domain-shift robustness

### 🚩 Red Flag 7: Reconstruction Loss Alone May Be Insufficient

**Issue:** The proposal mentions "reconstruction and contrastive losses" but offers no details on the contrastive formulation, temperature parameter, or weighting scheme.

**Mitigation:**
- **Specify Loss Function:**
  ```
  L_total = α * L_recon + β * L_contrastive + γ * L_kld
  ```
  where:
  - L_recon = MSE(g_t, g_hat_t) (reconstruction of histogram features)
  - L_contrastive = InfoNCE loss with τ = 0.07 (temperature)
  - L_kld = KL divergence regularization on GRU hidden states
  - α, β, γ: learnable or fixed (requires ablation)
- **Justification:** Why contrastive learning?
  - Answer: Encourages normal histogram evolution to cluster in latent space; anomalies repelled
- **Ablation:** Train with only L_recon; compare AUC loss if contrastive is added

### 🚩 Red Flag 8: No Discussion of Class Imbalance

**Issue:** Anomalous frames are rare (typically 10–20% of test frames). No discussion of weighted sampling or focal loss.

**Mitigation:**
- **In training:** Use weighted random sampling:
  - Sample 50% normal frames, 50% frames from epochs with anomalies (synthetic or from test set inference)
  - Ensures balanced mini-batches
- **In loss:** Optionally apply focal loss or class weights:
  ```
  L_weighted = α * L_recon  (with dynamic weighting based on frame type)
  ```
- **Evaluation:** Report precision-recall curve explicitly; don't rely on AUC alone

### 🚩 Red Flag 9: GRU Memory Dimension Unspecified

**Issue:** The proposal mentions GRU for memory but doesn't specify hidden dimension, number of layers, or why GRU over LSTM.

**Mitigation:**
- **Specify Architecture:**
  - GRU hidden size: 64 (balance expressiveness vs. memory)
  - GRU layers: 2 (deeper message passing)
  - Why GRU over LSTM: Fewer parameters; faster training on modest datasets
- **Ablation:** Try LSTM (2 layers, 64 hidden) and compare (expect <1% AUC difference)
- **Hyperparameter sweep:** Test hidden sizes [32, 64, 128] via Optuna

### 🚩 Red Flag 10: No Handling of Variable-Length Videos

**Issue:** Datasets have variable frame counts per video (UCSD: 198 frames; Avenue: variable). Proposal doesn't address padding or dynamic sequence handling.

**Mitigation:**
- **Data Pipeline:**
  - Chunk long videos into overlapping windows of T frames (e.g., T=32 for UCSD, T=50 for ShanghaiTech)
  - Use stride of T/2 for training (ensures coverage without excessive redundancy)
  - At test time, use stride=T/4 for fine-grained frame-level scores
- **Batching:** Use PyTorch's `DataLoader` with `collate_fn` to handle variable-length batches
- **Padding:** For shorter videos, pad with repeated frames (normal pattern); exclude padding frames from AUC calculation

### 🚩 Red Flag 11: Unclear Optimal Hyperparameter Space

**Issue:** Proposal mentions Adam (10⁻³ learning rate) and batch size 32, but no justification or search range.

**Mitigation:**
- **Define Hyperparameter Search Space:**
  ```
  learning_rate: [1e-4, 1e-3, 5e-3]
  batch_size: [16, 32, 64]
  histogram_bins: [16, 32, 64]
  temporal_window: [16, 32, 64]
  gnn_layers: [1, 2, 3]
  gru_hidden: [32, 64, 128]
  α (recon weight): [0.5, 1.0, 2.0]
  β (contrastive weight): [0.1, 0.5, 1.0]
  ```
- **Optimization:** Use Optuna with 200 trials; early stopping on validation AUC plateau
- **Report:** Best hyperparameters in final results; document sensitivity analysis

### 🚩 Red Flag 12: Insufficient Ablation Study Scope

**Issue:** Proposal mentions ablation on "histogram bin count, temporal window size, and GNN depth" but provides no plan for systematic execution.

**Mitigation:**
- **Mandatory Ablation Table (in final report):**
  | Component | Variant | UCSD Ped2 AUC | Change |
  |-----------|---------|---------------|--------|
  | Feature | HSV only | 80.2% | Baseline |
  | " | +RGB | 82.1% | +1.9% |
  | " | +OptFlow | 84.3% | +2.2% |
  | " | +LBP | 84.5% | +0.2% |
  | Histogram bins | 16 | 83.0% | -1.5% |
  | " | 32 | 84.5% | Baseline |
  | " | 64 | 84.2% | -0.3% |
  | Temporal window | 16 | 82.1% | -2.4% |
  | " | 32 | 84.5% | Baseline |
  | " | 64 | 83.9% | -0.6% |
  | GNN depth | 1 layer | 82.0% | -2.5% |
  | " | 2 layers | 84.5% | Baseline |
  | " | 3 layers | 84.1% | -0.4% |
  | Edge weight | Euclidean | 81.2% | -3.3% |
  | " | Sinkhorn EMD | 84.5% | Baseline |
  | Loss | Recon only | 81.0% | -3.5% |
  | " | +Contrastive | 84.5% | Baseline |

- **Interpretation:** Each row answers "why is this design choice necessary?"

---

## 7. Implementation Roadmap (Revised)

### Phase 1: Setup & Data Pipeline (Weeks 1–2)

**Deliverables:**
- Environment setup (PyTorch 2.0+, PyTorch Geometric, OpenCV, Optuna, W&B)
- Data loading pipeline for UCSD Ped2, ShanghaiTech, Avenue
- Dataset statistics report (frame counts, anomaly proportions, frame rate consistency)
- Baseline histogram extraction code (no learning yet)
- **Output:** `data/splits.json` with train/val/test indices; `src/data_loader.py`

### Phase 2: Feature & Graph Construction (Weeks 3–4)

**Deliverables:**
- Multi-channel feature extraction (HSV, RGB, LBP, optical flow)
- **Sinkhorn EMD implementation** (CORE, not stretch)
  - Benchmark: 1000 EMD computations on 187-dim histograms
  - Target: ≤5 ms per EMD (vs. >50 ms for standard)
- Temporal graph construction and serialization (PyTorch Geometric format)
- Graph statistics visualization (node count, edge weight distribution)
- **Output:** `src/features.py`, `src/graph_builder.py`, `graphs/` with sample visualizations

### Phase 3: Model Architecture & Training (Weeks 5–7)

**Deliverables:**
- Temporal Graph Network model with GRU memory
  - Specify: hidden dimensions, number of layers, activation functions
  - Document: Why GRU over LSTM?
- Loss functions:
  - Reconstruction loss (MSE)
  - Contrastive loss (InfoNCE with τ = 0.07)
  - Optional: KL divergence on latent representations
- Training loop with early stopping (validation AUC plateau)
- W&B logging: loss curves, AUC curves, learning rate schedule
- **Output:** `src/model.py`, `src/train.py`, W&B run traces

### Phase 4: Evaluation & Baselines (Weeks 8–9)

**Deliverables:**
- Evaluation harness (frame-level AUC, video-level AUC, per-anomaly-type AUC)
- Baseline implementations:
  1. Simple histogram autoencoder (own code)
  2. Re-implement or port Shi et al. (2023) if possible
  3. Optical flow + Gaussian Mixture Models
  4. Variational Autoencoder (VAE)
- Comparison table: HistoGraph vs. baselines on UCSD Ped2
- Statistical significance testing (McNemar's test)
- **Output:** `src/eval.py`, `results/baseline_comparison.csv`, `results/significance_tests.md`

### Phase 5: Ablation Studies (Weeks 10–11)

**Deliverables:**
- Systematic ablation (feature components, histogram bins, temporal windows, GNN depth, loss weighting)
- Cross-dataset evaluation (train UCSD Ped2, test ShanghaiTech/Avenue)
- Domain generalization analysis (zero-shot transfer performance drop)
- **Output:** `results/ablation_table.csv`, `results/cross_dataset_analysis.md`

### Phase 6: Optimization & Finalization (Weeks 12–14)

**Deliverables:**
- Latency/memory profiling (achieve ≤100 ms/frame on CPU)
- Qualitative heatmaps on AVA and ShanghaiTech
- Final results compilation (AUC on all datasets, statistical CIs)
- Code cleanup, documentation, and reproducibility checklist
- Final report with all ablations, baselines, and comparisons
- **Output:** `results/final_report.md`, model checkpoint, reproducibility guide

---

## 8. Success Criteria Checklist

- [ ] HistoGraph AUC on UCSD Ped2 ≥ 83% (minimum 0% improvement over Shi et al. baseline)
- [ ] HistoGraph AUC on UCSD Ped2 ≥ 88% (target 5% improvement)
- [ ] Improvement is statistically significant (p < 0.05, McNemar's test)
- [ ] Cross-dataset transfer: zero-shot AUC on ShanghaiTech ≥ 80%
- [ ] Inference latency ≤ 100 ms/frame on CPU (Intel i7 equivalent)
- [ ] Model size ≤ 50 MB
- [ ] Ablation table completed and published
- [ ] At least 3 baseline comparisons executed
- [ ] Qualitative heatmaps generated and visualized
- [ ] All code reproducible (random seeds set, data splits documented, hyperparameters logged)
- [ ] 95% confidence intervals reported for all primary metrics
- [ ] No red flags left unaddressed in final implementation

---

## 9. Tools and Dependencies

```
Core Libraries:
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- OpenCV 4.8
- NumPy, Pandas, Scikit-learn 1.3
- Optuna 3.2 (hyperparameter tuning)
- Weights & Biases (W&B) for experiment tracking

Optimization:
- POT (Python Optimal Transport) for Wasserstein/EMD
- Numba or Cython for performance-critical feature extraction

Visualization & Reporting:
- Matplotlib, Seaborn
- NetworkX (graph visualization)
- Jupyter Notebooks for analysis

Version Control:
- Git, GitHub (document all commits with clear messages)
```

---

## 10. Communication and Deliverables

**Weekly Milestones:**
- Every Friday: W&B dashboard update with latest metrics
- Every two weeks: Internal sync on blockers and design decisions
- Final report: Comprehensive PDF with all tables, ablations, and visualizations

**Final Deliverables:**
1. Implementation code (GitHub repo with README)
2. Trained model checkpoint (state_dict)
3. Final report (PDF, 15–20 pages)
4. Reproducibility kit: scripts, data splits, hyperparameters
5. Qualitative analysis: heatmap videos, cross-dataset visualizations

---

**End of Implementation Plan**