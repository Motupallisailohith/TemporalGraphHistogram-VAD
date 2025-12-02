#!/usr/bin/env python3
"""
NPY/NPZ File Viewer and Analyzer
Interactive tool to view, analyze, and visualize .npy and .npz files
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import argparse

def analyze_npz_file(file_path):
    """Analyze an NPZ file (compressed archive with multiple arrays)"""
    print(f"\n{'='*60}")
    print(f"ANALYZING NPZ: {file_path}")
    print(f"{'='*60}")
    
    try:
        # Load the NPZ file
        npz_data = np.load(file_path)
        
        # Get list of arrays in the archive
        array_names = list(npz_data.files)
        print(f"📦 Archive Contents: {len(array_names)} arrays")
        print(f"   Array names: {array_names}")
        
        # Analyze each array in the archive
        all_arrays = {}
        for name in array_names:
            print(f"\n📊 Array: '{name}'")
            data = npz_data[name]
            all_arrays[name] = data
            
            print(f"   Shape: {data.shape}")
            print(f"   Data type: {data.dtype}")
            print(f"   Size: {data.size} elements")
            print(f"   Memory: {data.nbytes / 1024:.2f} KB")
            
            # Statistical analysis
            if data.size > 0:
                print(f"   Min: {np.min(data):.6f}")
                print(f"   Max: {np.max(data):.6f}")
                print(f"   Mean: {np.mean(data):.6f}")
                print(f"   Std: {np.std(data):.6f}")
                
                # Show sample values for smaller arrays
                if len(data.shape) == 1 and data.shape[0] <= 20:
                    print(f"   Values: {data}")
                elif len(data.shape) == 1:
                    print(f"   First 5: {data[:5]}")
                    print(f"   Last 5: {data[-5:]}")
        
        npz_data.close()
        return all_arrays
        
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

def analyze_npy_file(file_path):
    """Analyze a single NPY file"""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {file_path}")
    print(f"{'='*60}")
    
    try:
        # Load the array
        data = np.load(file_path)
        
        # Basic information
        print(f"📊 Basic Information:")
        print(f"   Shape: {data.shape}")
        print(f"   Data type: {data.dtype}")
        print(f"   Size: {data.size} elements")
        print(f"   Memory usage: {data.nbytes / 1024:.2f} KB")
        
        # Statistical analysis
        if data.size > 0:
            print(f"\n📈 Statistical Summary:")
            print(f"   Min: {np.min(data):.6f}")
            print(f"   Max: {np.max(data):.6f}")
            print(f"   Mean: {np.mean(data):.6f}")
            print(f"   Std: {np.std(data):.6f}")
            print(f"   Median: {np.median(data):.6f}")
            
            # For 1D arrays, show first and last few values
            if len(data.shape) == 1:
                print(f"\n🔍 Sample Values:")
                if data.shape[0] <= 20:
                    print(f"   All values: {data}")
                else:
                    print(f"   First 10: {data[:10]}")
                    print(f"   Last 10: {data[-10:]}")
            
            # For 2D arrays, show shape info and sample
            elif len(data.shape) == 2:
                print(f"\n📋 2D Array Info:")
                print(f"   Rows: {data.shape[0]}, Columns: {data.shape[1]}")
                if data.shape[0] <= 10 and data.shape[1] <= 10:
                    print(f"   Full array:\n{data}")
                else:
                    print(f"   Sample (top-left 5x5):\n{data[:5, :5]}")
            
            # For higher dimensions
            elif len(data.shape) > 2:
                print(f"\n📦 Multi-dimensional Array:")
                print(f"   Dimensions: {len(data.shape)}")
                print(f"   Shape breakdown: {' × '.join(map(str, data.shape))}")
                print(f"   Sample shape: {data.flatten()[:10]}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

def visualize_npz_data(all_arrays, file_path, save_plots=False):
    """Create visualizations for NPZ data (multiple arrays)"""
    if not all_arrays:
        return
    
    print(f"\n🎨 Creating NPZ visualizations...")
    
    num_arrays = len(all_arrays)
    if num_arrays == 0:
        return
    
    # Create subplots for all arrays
    cols = min(3, num_arrays)
    rows = (num_arrays + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, (name, data) in enumerate(all_arrays.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Visualize based on data dimensionality
        if len(data.shape) == 1:
            ax.plot(data, linewidth=1, alpha=0.8)
            ax.set_title(f'{name}\nShape: {data.shape}')
            ax.set_xlabel('Index')
            ax.set_ylabel('Value')
        elif len(data.shape) == 2:
            im = ax.imshow(data, cmap='viridis', aspect='auto')
            ax.set_title(f'{name}\nShape: {data.shape}')
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
            plt.colorbar(im, ax=ax)
        else:
            # For higher dimensions, plot flattened histogram
            flattened = data.flatten()
            ax.hist(flattened[:10000], bins=50, alpha=0.7, edgecolor='black')  # Sample for performance
            ax.set_title(f'{name}\nShape: {data.shape} (Distribution)')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
        
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(num_arrays, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_plots:
        plot_file = Path(file_path).parent / f"{Path(file_path).stem}_npz_visualization.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"   NPZ plot saved: {plot_file}")
    
    plt.show()

def visualize_npy_data(data, file_path, save_plots=False):
    """Create visualizations for NPY data"""
    if data is None:
        return
    
    print(f"\n🎨 Creating visualizations...")
    
    # Determine number of subplots needed
    num_plots = 0
    plots = []
    
    # 1D data - line plot and histogram
    if len(data.shape) == 1:
        num_plots = 2
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Line plot
        axes[0].plot(data, linewidth=1, color='blue', alpha=0.8)
        axes[0].set_title(f'Line Plot - {Path(file_path).name}')
        axes[0].set_xlabel('Index')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)
        
        # Histogram
        axes[1].hist(data, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1].set_title(f'Histogram - {Path(file_path).name}')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
    
    # 2D data - heatmap and statistics
    elif len(data.shape) == 2:
        num_plots = 2
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Heatmap
        im = axes[0].imshow(data, cmap='viridis', aspect='auto')
        axes[0].set_title(f'Heatmap - {Path(file_path).name}')
        axes[0].set_xlabel('Column Index')
        axes[0].set_ylabel('Row Index')
        plt.colorbar(im, ax=axes[0])
        
        # Flattened histogram
        axes[1].hist(data.flatten(), bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1].set_title(f'Value Distribution - {Path(file_path).name}')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
    
    # Higher dimensional - flatten and show distribution
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        flattened = data.flatten()
        ax.hist(flattened, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax.set_title(f'Value Distribution - {Path(file_path).name}\n(Flattened {len(data.shape)}D array)')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plot_file = Path(file_path).parent / f"{Path(file_path).stem}_visualization.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"   Plot saved: {plot_file}")
    
    plt.show()

def find_numpy_files(directory):
    """Find all NPY and NPZ files in a directory"""
    directory = Path(directory)
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return []
    
    npy_files = list(directory.rglob("*.npy"))
    npz_files = list(directory.rglob("*.npz"))
    all_files = npy_files + npz_files
    return sorted(all_files)

def interactive_mode():
    """Interactive mode for browsing NPY files"""
    print("🔍 Interactive NPY File Viewer")
    print("=" * 40)
    
    while True:
        print(f"\nOptions:")
        print("1. Analyze single file")
        print("2. Browse directory")
        print("3. Compare multiple files")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            file_path = input("Enter NPY/NPZ file path: ").strip().strip('"')
            if Path(file_path).exists():
                if Path(file_path).suffix == '.npz':
                    data = analyze_npz_file(file_path)
                    if data is not None:
                        show_viz = input("Show visualization? (y/n): ").strip().lower()
                        if show_viz in ['y', 'yes']:
                            save_viz = input("Save plots? (y/n): ").strip().lower() == 'y'
                            visualize_npz_data(data, file_path, save_viz)
                else:
                    data = analyze_npy_file(file_path)
                    if data is not None:
                        show_viz = input("Show visualization? (y/n): ").strip().lower()
                        if show_viz in ['y', 'yes']:
                            save_viz = input("Save plots? (y/n): ").strip().lower() == 'y'
                            visualize_npy_data(data, file_path, save_viz)
            else:
                print(f"❌ File not found: {file_path}")
        
        elif choice == '2':
            dir_path = input("Enter directory path: ").strip().strip('"')
            numpy_files = find_numpy_files(dir_path)
            
            if numpy_files:
                print(f"\n📁 Found {len(numpy_files)} NumPy files (NPY/NPZ):")
                for i, file in enumerate(numpy_files[:20], 1):  # Show first 20
                    file_type = "NPZ" if file.suffix == ".npz" else "NPY"
                    print(f"   {i:2d}. [{file_type}] {file}")
                
                if len(numpy_files) > 20:
                    print(f"   ... and {len(numpy_files) - 20} more files")
                
                try:
                    file_num = int(input(f"\nSelect file number (1-{min(20, len(numpy_files))}): "))
                    if 1 <= file_num <= min(20, len(numpy_files)):
                        selected_file = numpy_files[file_num - 1]
                        
                        # Handle NPZ or NPY files
                        if selected_file.suffix == '.npz':
                            data = analyze_npz_file(selected_file)
                            if data is not None:
                                show_viz = input("Show visualization? (y/n): ").strip().lower()
                                if show_viz in ['y', 'yes']:
                                    visualize_npz_data(data, selected_file)
                        else:
                            data = analyze_npy_file(selected_file)
                            if data is not None:
                                show_viz = input("Show visualization? (y/n): ").strip().lower()
                                if show_viz in ['y', 'yes']:
                                    visualize_npy_data(data, selected_file)
                except ValueError:
                    print("❌ Invalid selection")
            else:
                print("❌ No NumPy files found")
        
        elif choice == '3':
            print("📊 Compare Multiple Files")
            files = []
            while True:
                file_path = input("Enter NPY file path (or 'done' to finish): ").strip()
                if file_path.lower() == 'done':
                    break
                if Path(file_path).exists():
                    files.append(file_path)
                    print(f"   Added: {Path(file_path).name}")
                else:
                    print(f"   ❌ File not found: {file_path}")
            
            if len(files) >= 2:
                compare_npy_files(files)
        
        elif choice == '4':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")

def compare_npy_files(file_paths):
    """Compare multiple NPY files"""
    print(f"\n📊 Comparing {len(file_paths)} files:")
    
    data_list = []
    valid_files = []
    
    for file_path in file_paths:
        try:
            data = np.load(file_path)
            data_list.append(data)
            valid_files.append(file_path)
            print(f"✅ {Path(file_path).name}: {data.shape} {data.dtype}")
        except Exception as e:
            print(f"❌ {Path(file_path).name}: Error - {e}")
    
    if len(data_list) < 2:
        print("❌ Need at least 2 valid files for comparison")
        return
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, len(data_list), figsize=(4*len(data_list), 8))
    if len(data_list) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (data, file_path) in enumerate(zip(data_list, valid_files)):
        # Top row: data visualization
        if len(data.shape) == 1:
            axes[0, i].plot(data, linewidth=1)
            axes[0, i].set_title(f'{Path(file_path).name}\nShape: {data.shape}')
        else:
            if len(data.shape) == 2:
                im = axes[0, i].imshow(data, cmap='viridis', aspect='auto')
                plt.colorbar(im, ax=axes[0, i])
            else:
                axes[0, i].plot(data.flatten()[:1000])  # Sample for high-dim
            axes[0, i].set_title(f'{Path(file_path).name}\nShape: {data.shape}')
        
        # Bottom row: histograms
        flattened = data.flatten()
        axes[1, i].hist(flattened, bins=30, alpha=0.7, edgecolor='black')
        axes[1, i].set_title(f'Distribution\nMin:{np.min(data):.3f} Max:{np.max(data):.3f}')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="NPY File Viewer and Analyzer")
    parser.add_argument("file", nargs="?", help="NPY file to analyze")
    parser.add_argument("-d", "--directory", help="Directory to browse for NPY files")
    parser.add_argument("-v", "--visualize", action="store_true", help="Show visualizations")
    parser.add_argument("-s", "--save-plots", action="store_true", help="Save plot images")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive or (not args.file and not args.directory):
        interactive_mode()
    elif args.file:
        if Path(args.file).suffix == '.npz':
            data = analyze_npz_file(args.file)
            if data is not None and args.visualize:
                visualize_npz_data(data, args.file, args.save_plots)
        else:
            data = analyze_npy_file(args.file)
            if data is not None and args.visualize:
                visualize_npy_data(data, args.file, args.save_plots)
    elif args.directory:
        numpy_files = find_numpy_files(args.directory)
        print(f"\n📁 Found {len(numpy_files)} NumPy files in {args.directory}")
        for file in numpy_files:
            if file.suffix == '.npz':
                analyze_npz_file(file)
            else:
                analyze_npy_file(file)

if __name__ == "__main__":
    main()