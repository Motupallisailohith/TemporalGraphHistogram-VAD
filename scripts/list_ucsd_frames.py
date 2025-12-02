import os

def get_frame_paths(seq_folder):
    frame_files = [f for f in os.listdir(seq_folder) if f.endswith('.bmp') or f.endswith('.tif')]
    frame_files = sorted(frame_files)
    frame_paths = [os.path.join(seq_folder, f) for f in frame_files]
    return frame_paths

# Example usage for Test001 (adjust this path for your actual folder!):
seq_folder = 'data/raw/UCSD_Ped2/UCSDped2/Test/Test001'
frames = get_frame_paths(seq_folder)
print(frames[:5])  # Shows first 5 frame paths
