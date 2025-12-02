import numpy as np
import json

# Example: Test001
histograms = np.load('data/raw/UCSD_Ped2/UCSDped2/Test_histograms/Test001_histograms.npy')
with open('data/splits/ucsd_ped2_labels.json') as f:
    labels = json.load(f)["Test001"]
print('Num histograms:', len(histograms))
print('Num labels:', len(labels))
assert len(histograms) == len(labels), "Mismatch between features and labels!"
