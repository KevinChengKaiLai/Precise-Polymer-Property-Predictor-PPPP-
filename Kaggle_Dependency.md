# Kaggle Environment Setup & Dependency Guide

This guide explains how to set up the environment for **AIMNet-X2D** inference on Kaggle.

Because this is a **Code Competition**, internet access is disabled during submission. Standard `pip install` commands will fail. Furthermore, the `AIMNet` codebase has complex dependencies (PyTorch Geometric, RDKit) that require specific compilation steps to run on Kaggle's Linux environment.

Follow these steps exactly to ensure reproducibility.


##  Prerequisites

1.  **Kaggle Notebook Settings:**
    * **Accelerator:** GPU T4 x2 (Required for compiling CUDA extensions like `torch_scatter`).
    * **Internet:** OFF (Required for submission).
    * **Persistence:** Files in `/kaggle/working` are temporary. We use `/kaggle/working/external_libs` to install libraries fresh for every run.



##  Step 1: Prepare Dependencies (Local Computer)

You cannot download packages directly on Kaggle. You must download them on your local machine (Mac/Windows/Linux) and upload them as a Kaggle Dataset.

**Crucial Note:** You must force `pip` to download the **Linux** versions of the packages, even if you are on a Mac or Windows.

### 1. Create a clean folder locally
Open your terminal/command prompt:

```bash
mkdir kaggle-submission
cd kaggle-submission
````

### 2\. Download RDKit (Binary Wheel)

We download the specific Linux binary wheel for RDKit to avoid long compilation times.

```bash
pip download rdkit==2024.3.2 \
  --dest . \
  --platform manylinux_2_17_x86_64 \
  --only-binary=:all: \
  --python-version 3.11 \
  --no-deps
```

### 3\. Download PyTorch Geometric Extensions (Source Code)

We download the **source code** (`.tar.gz`) for PyTorch Geometric extensions. This allows Kaggle to compile them specifically for its own GPU/CUDA version (which avoids "version mismatch" errors).

```bash
pip download \
  mhfp \
  torch_geometric \
  torch_scatter \
  torch_sparse \
  torch_cluster \
  torch_spline_conv \
  --dest . \
  --no-binary=:all: \
  --no-deps
```

### 4\. Verify Files

Your folder should now contain:

  * `rdkit-....whl` (Binary)
  * `torch_scatter-....tar.gz` (Source)
  * `torch_sparse-....tar.gz` (Source)
  * ...and others.

### 5\. Upload to Kaggle

1.  Go to [Kaggle Datasets](https://www.kaggle.com/datasets).
2.  Click **New Dataset**.
3.  Drag and drop your `kaggle-submission` folder.
4.  Name the dataset: `kaggle-submission` (or similar).
5.  **Create**.



##  Step 2: Install in Notebook (Offline)

Add this **Master Setup Cell** to the very top of your submission notebook. It installs the packages from your uploaded dataset.

**What this script does:**

1.  **Installs RDKit** from the binary wheel.
2.  **Compiles PyTorch Extensions** from source (using Kaggle's GPU).
      * *Note:* It copies the source files to a writable temp folder first to avoid "Read-only file system" errors.
      * *Note:* It uses `--no-build-isolation` to fix the "missing flit\_core" error.
3.  **Sets up PyTorch Geometric** in a custom folder (`/kaggle/working/external_libs`) to prevent path conflicts.



```python
import sys
import subprocess
import os
import glob
import shutil

# CONFIGURATION
# Update this path if your dataset name is different
INPUT_PATH = '/kaggle/input/kaggle-submission'
FINAL_LIB_PATH = '/kaggle/working/external_libs'
os.makedirs(FINAL_LIB_PATH, exist_ok=True)

def find_package_path(base_name):
    search_pattern = os.path.join(INPUT_PATH, f"{base_name}*")
    matches = glob.glob(search_pattern)
    # Filter out the dataset root folder itself if matched
    matches = [m for m in matches if os.path.basename(m) != "kaggle-submission"]
    return matches[0] if matches else None

def install_dependencies():
    print("🚀 Starting Dependency Installation...")
    
    # 1. Install RDKit (Binary Wheel)
    rdkit_path = find_package_path("rdkit")
    if rdkit_path:
        try:
            import rdkit
            print("✅ RDKit is already installed.")
        except ImportError:
            print("Installing RDKit...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", rdkit_path, "--no-deps", "--no-index"])
    
    # 2. Install PyTorch Extensions (Source Compilation)
    # Order matters! Dependencies first.
    pkgs = ["mhfp", "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv"]
    
    for pkg in pkgs:
        try:
            __import__(pkg)
            print(f"✅ {pkg} is already installed.")
            continue
        except ImportError:
            pass

        print(f"Installing {pkg} (Compiling from source)...")
        pkg_path = find_package_path(pkg)
        if pkg_path:
            # Copy to writable temp folder to avoid Read-Only errors
            temp_build = f"/kaggle/working/build_{pkg}"
            if os.path.exists(temp_build): shutil.rmtree(temp_build)
            shutil.copytree(pkg_path, temp_build)
            
            # Handle potential double-nesting of source folders (common with .tar.gz)
            setup_dir = temp_build
            if not os.path.exists(os.path.join(temp_build, 'setup.py')):
                for child in os.listdir(temp_build):
                    if os.path.isdir(os.path.join(temp_build, child)):
                        setup_dir = os.path.join(temp_build, child)
                        break
            
            # CRITICAL: --no-build-isolation prevents pip from trying to download build tools from internet
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                setup_dir, "--no-deps", "--no-index", "--no-build-isolation"
            ])

def setup_torch_geometric():
    print("\n📦 Setting up PyTorch Geometric...")
    target_path = os.path.join(FINAL_LIB_PATH, 'torch_geometric')
    
    if os.path.exists(target_path):
        print("✅ PyG is already set up.")
        return

    # Move library code manually to avoid complex build errors
    pkg_path = find_package_path("torch_geometric")
    if not pkg_path: return

    temp_extract = '/kaggle/working/temp_pyg_extract'
    if os.path.exists(temp_extract): shutil.rmtree(temp_extract)
    shutil.copytree(pkg_path, temp_extract)

    # Locate the inner library folder
    inner_lib = None
    for root, dirs, files in os.walk(temp_extract):
        if 'torch_geometric' in dirs and os.path.exists(os.path.join(root, 'torch_geometric', '__init__.py')):
            inner_lib = os.path.join(root, 'torch_geometric')
            break
    
    if inner_lib:
        shutil.copytree(inner_lib, target_path)
        print(f"✅ Moved torch_geometric to: {target_path}")
    
    shutil.rmtree(temp_extract)

if __name__ == "__main__":
    install_dependencies()
    setup_torch_geometric()
    # Add to path so subsequent cells can import it
    if FINAL_LIB_PATH not in sys.path:
        sys.path.append(FINAL_LIB_PATH)
    print("\n🎉 Environment Setup Complete!")
```



## Step 3: Run Inference (Launcher Script)

Because of name collisions between the AIMNet `datasets` folder and the HuggingFace `datasets` library, **DO NOT** run `import main` directly in the notebook.

Instead, create a `launcher.py` script that configures the system path **before** any imports occur.

**In a notebook cell:**

```python
%%writefile /kaggle/working/launcher.py
import sys
import os

# 1. SETUP PATHS (Priority is Key)
# Force Python to look in our custom folders FIRST
aimnet_src = '/kaggle/input/aimnet-x2d-code/AIMNet-X2D/src'
sys.path.insert(0, aimnet_src)

ext_lib = '/kaggle/working/external_libs'
sys.path.insert(0, ext_lib)

# 2. RUN PIPELINE
from main import main_runner, parse_main_arguments
# ... (Your inference code here) ...
```

**Execute it with:**

```bash
!python /kaggle/working/launcher.py
```
