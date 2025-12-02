# Component 4: GNN Training & Scoring

## 🎯 Overview

**This is your CORE INNOVATION** - Graph Neural Network autoencoder for temporal video anomaly detection!

Component 4 implements a GNN-based approach that learns normal temporal patterns from training sequences and detects anomalies through reconstruction error.

**Expected Performance:**
- **Baseline L2**: ~51% AUC (current)
- **GNN Method**: ~85% AUC (target)
- **Innovation**: Captures temporal graph patterns, not just per-frame features

---

## 📦 What's Included

### Scripts Created

1. **`extract_train_cnn_features.py`**
   - Extracts CNN features for training sequences (Train001-Train016)
   - Uses same ResNet50 architecture as test sequences
   - Output: `Train*_cnn_features.npy` (2048-dim per frame)

2. **`generate_training_graphs.py`**
   - Builds temporal graphs from training CNN features
   - Same window_k=2 and cosine similarity weighting as test
   - Output: `Train*_temporal_graph.npz`

3. **`phase3_4a_train_gnn.py`** ⭐ **CORE COMPONENT**
   - Trains GNN autoencoder on normal sequences
   - Architecture: 2048 → 512 → 128 → 512 → 2048 (GCN layers)
   - Loss: MSE reconstruction error
   - Training: 50 epochs (~15-20 min on GPU)
   - Output: `models/gnn_autoencoder.pth`

4. **`phase3_4b_score_gnn.py`** ⭐ **CORE COMPONENT**
   - Scores test sequences using trained GNN
   - High reconstruction error = anomaly
   - Output: `Test*_gnn_scores.npy` (one score per frame)

5. **`evaluate_gnn_scores.py`**
   - Computes ROC/AUC metrics for GNN
   - Compares GNN vs Baseline L2
   - Generates ROC curve plots
   - Output: `gnn_evaluation_results.json`

6. **`run_component4_pipeline.py`** 🚀 **ONE-CLICK SOLUTION**
   - Runs complete Component 4 pipeline
   - All 4 steps automated
   - Estimated time: 15-25 minutes

---

## 🚀 Quick Start

### Option 1: Run Complete Pipeline (Recommended)

```bash
python scripts/run_component4_pipeline.py
```

This will execute all 4 steps automatically:
1. Extract training CNN features (~2 min)
2. Build training temporal graphs (~3 sec)
3. Train GNN autoencoder (~15-20 min)
4. Score test sequences (~1 min)

**Total time: ~20-25 minutes**

---

### Option 2: Run Steps Individually

#### Step 1: Extract Training CNN Features
```bash
python scripts/extract_train_cnn_features.py
```

**Output:**
- `data/processed/cnn_features/Train001-Train016_cnn_features.npy`
- Each file: (num_frames, 2048) - ResNet50 features

#### Step 2: Build Training Temporal Graphs
```bash
python scripts/generate_training_graphs.py
```

**Output:**
- `data/processed/temporal_graphs/Train001-Train016_temporal_graph.npz`
- Each file contains: node_features, adjacency_matrix, edge_index

#### Step 3: Train GNN Autoencoder
```bash
python scripts/phase3_4a_train_gnn.py
```

**Output:**
- `models/gnn_autoencoder.pth` (final model)
- `models/gnn_autoencoder_best.pth` (best epoch)
- `models/training_history.json` (loss curves)

**Training Details:**
- Architecture: 2048 → 512 → 128 → 512 → 2048
- Loss: MSE reconstruction error
- Epochs: 50
- Optimizer: Adam (lr=0.001)
- GPU accelerated (RTX 4050)

#### Step 4: Score Test Sequences
```bash
python scripts/phase3_4b_score_gnn.py
```

**Output:**
- `data/processed/gnn_scores/Test001-Test012_gnn_scores.npy`
- Each file: (num_frames,) - Anomaly scores per frame

#### Step 5: Evaluate Performance
```bash
python scripts/evaluate_gnn_scores.py
```

**Output:**
- `data/processed/evaluation_results/gnn_evaluation_results.json`
- `reports/gnn_vs_baseline_roc.png` (ROC curve plot)

**Metrics:**
- Overall AUC (all sequences combined)
- Per-sequence AUC
- Comparison vs baseline L2

---

## 🧠 Architecture Details

### GNN Autoencoder

```
Input: Temporal Graph
  ├── Node features: (num_frames, 2048) - CNN features
  └── Edge index: (2, num_edges) - Temporal connections

Encoder (Compress):
  ├── GCNConv(2048 → 512) + ReLU
  └── GCNConv(512 → 128) + ReLU
  
Latent Space: (num_frames, 128)

Decoder (Reconstruct):
  ├── GCNConv(128 → 512) + ReLU
  └── GCNConv(512 → 2048)

Output: Reconstructed node features (num_frames, 2048)

Loss: MSE(original, reconstructed)
```

### Key Innovation

**Traditional methods (baseline):**
- Compare each frame to historical average
- No temporal context
- Per-frame analysis only

**GNN method (Component 4):**
- Learns temporal graph patterns
- Message passing between temporally connected frames
- Captures evolution of features over time
- Anomalies = patterns that don't fit learned normal behavior

---

## 📊 Expected Results

### Training Sequences (Normal Behavior)
- Train001-Train016: All normal pedestrian behavior
- GNN learns to reconstruct these patterns with low error

### Test Sequences (Mixed)
- Normal frames: Low reconstruction error (GNN recognizes pattern)
- Anomaly frames: High reconstruction error (unfamiliar pattern)

### Performance Targets
| Method | AUC | Description |
|--------|-----|-------------|
| Random | 50.0% | Baseline (chance) |
| L2 Distance | 51.3% | Current baseline |
| **GNN (Target)** | **~85%** | **Expected with proper training** |
| SOTA | 90%+ | State-of-the-art methods |

---

## 🔍 Troubleshooting

### Issue: "No training graphs found"
**Solution:**
```bash
# First extract CNN features for training
python scripts/extract_train_cnn_features.py

# Then build training graphs
python scripts/generate_training_graphs.py
```

### Issue: "Model not found"
**Solution:**
```bash
# Train the GNN model first
python scripts/phase3_4a_train_gnn.py
```

### Issue: Training is slow
**Check GPU usage:**
```bash
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```
- Should print `GPU: True`
- If False, training will use CPU (much slower)

### Issue: Out of memory during training
**Reduce batch size or model size:**
- Edit `phase3_4a_train_gnn.py`
- Change `hidden_dim=512` to `hidden_dim=256`
- Change `latent_dim=128` to `latent_dim=64`

---

## 📈 Monitoring Training

During training, you'll see output like:
```
Epoch   1/50 | Loss: 0.234567 | Best: 0.234567 (epoch 1)
Epoch   5/50 | Loss: 0.123456 | Best: 0.123456 (epoch 5)
Epoch  10/50 | Loss: 0.089012 | Best: 0.089012 (epoch 10)
...
```

**Good signs:**
- Loss decreases over epochs
- Best loss improves steadily
- Final loss < 0.05 typically indicates good learning

**Warning signs:**
- Loss increases or stays flat
- Best loss doesn't improve after many epochs
- Final loss > 0.1 may indicate poor learning

---

## 🎯 Next Steps After Component 4

1. **Analyze Results:**
   - Check GNN vs Baseline comparison
   - Identify which sequences improved most
   - Visualize reconstruction errors

2. **Hyperparameter Tuning:**
   - Try different architectures (hidden_dim, latent_dim)
   - Experiment with learning rates
   - Adjust number of epochs

3. **Advanced Experiments:**
   - Add attention mechanisms (GAT instead of GCN)
   - Temporal convolutions (T-GCN)
   - Multi-modal fusion (combine with optical flow)

4. **Visualization Dashboard:**
   - Plot anomaly scores over time
   - Show attention weights on temporal connections
   - Compare all feature types side-by-side

---

## 📚 Dependencies

### Already Installed
- PyTorch 2.5.1+cu121
- torchvision
- numpy
- scikit-learn

### New Requirements (Installed)
- torch-geometric
- torch-scatter
- torch-sparse

---

## 🏆 Why This Matters

This is the **main contribution** of your thesis/project:

1. **Novel approach**: Graph-based temporal modeling for VAD
2. **Interpretable**: Reconstruction error directly indicates anomalies
3. **Scalable**: Handles variable-length sequences efficiently
4. **Generalizable**: Can extend to other video anomaly datasets

**Key insight:** Anomalies are not just "different" frames, but frames whose **temporal relationships** differ from normal patterns!

---

## 📞 Support

If you encounter issues:
1. Check prerequisites (GPU, training graphs, etc.)
2. Review error messages carefully
3. Try running individual steps to isolate problems
4. Check output files were generated correctly

---

## 🎉 Success Criteria

✅ **Component 4 is complete when:**
- All training graphs generated (16 sequences)
- GNN model trained successfully (loss converging)
- All test sequences scored (12 sequences)
- GNN AUC > Baseline AUC (~51%)
- Ideally: GNN AUC > 75%

**Ready to revolutionize video anomaly detection? Let's go!** 🚀
