# **Kaggle Environment Setup & Dependency Guide**

This guide explains how to set up the environment for **AIMNet-X2D** inference on Kaggle.

Because this is a **Code Competition**, internet access is disabled during submission. Standard pip install commands will fail. Furthermore, the AIMNet codebase has complex dependencies (PyTorch Geometric, RDKit) that require specific compilation steps to run on Kaggle's Linux environment.

Follow these steps exactly to ensure reproducibility.

## **🛑 Prerequisites**

1. **Kaggle Notebook Settings:**  
   * **Accelerator:** GPU T4 x2 (Required for compiling CUDA extensions like torch\_scatter).  
   * **Internet:** OFF (Required for submission).  
   * **Persistence:** Files in /kaggle/working are temporary. We use /kaggle/working/external\_libs to install libraries fresh for every run.

## **📦 Step 1: Prepare Dependencies (Local Computer)**

You cannot download packages directly on Kaggle. You must download them on your local machine (Mac/Windows/Linux) and upload them as a Kaggle Dataset.

**Crucial Note:** You must force pip to download the **Linux** versions of the packages, even if you are on a Mac or Windows.

### **1\. Create a clean folder locally**

Open your terminal/command prompt:

mkdir kaggle-submission  
cd kaggle-submission

### **2\. Download RDKit (Binary Wheel)**

We download the specific Linux binary wheel for RDKit to avoid long compilation times.

pip download rdkit==2024.3.2 \\  
  \--dest . \\  
  \--platform manylinux\_2\_17\_x86\_64 \\  
  \--only-binary=:all: \\  
  \--python-version 3.11 \\  
  \--no-deps

### **3\. Download PyTorch Geometric Extensions (Source Code)**

We download the **source code** (.tar.gz) for PyTorch Geometric extensions. This allows Kaggle to compile them specifically for its own GPU/CUDA version (which avoids "version mismatch" errors).

pip download \\  
  mhfp \\  
  torch\_geometric \\  
  torch\_scatter \\  
  torch\_sparse \\  
  torch\_cluster \\  
  torch\_spline\_conv \\  
  \--dest . \\  
  \--no-binary=:all: \\  
  \--no-deps

### **4\. Verify Files**

Your folder should now contain:

* rdkit-....whl (Binary)  
* torch\_scatter-....tar.gz (Source)  
* torch\_sparse-....tar.gz (Source)  
* ...and others.

### **5\. Upload to Kaggle**

1. Go to [Kaggle Datasets](https://www.kaggle.com/datasets).  
2. Click **New Dataset**.  
3. Drag and drop your kaggle-submission folder.  
4. Name the dataset: kaggle-submission (or similar).  
5. **Create**.

## **🛠️ Step 2: Install in Notebook (Offline)**

Add this **Master Setup Cell** to the very top of your submission notebook. It installs the packages from your uploaded dataset.

**What this script does:**

1. **Installs** RDKit from the binary wheel.  
2. **Compiles PyTorch Extensions** from source (using Kaggle's GPU).  
   * *Note:* It copies the source files to a writable temp folder first to avoid "Read-only file system" errors.  
   * *Note:* It uses \--no-build-isolation to fix the "missing flit\_core" error.  
3. **Sets up PyTorch Geometric** in a custom folder (/kaggle/working/external\_libs) to prevent path conflicts.

import sys  
import subprocess  
import os  
import glob  
import shutil

\# CONFIGURATION  
\# Update this path if your dataset name is different  
INPUT\_PATH \= '/kaggle/input/kaggle-submission'  
FINAL\_LIB\_PATH \= '/kaggle/working/external\_libs'  
os.makedirs(FINAL\_LIB\_PATH, exist\_ok=True)

def find\_package\_path(base\_name):  
    search\_pattern \= os.path.join(INPUT\_PATH, f"{base\_name}\*")  
    matches \= glob.glob(search\_pattern)  
    \# Filter out the dataset root folder itself if matched  
    matches \= \[m for m in matches if os.path.basename(m) \!= "kaggle-submission"\]  
    return matches\[0\] if matches else None

def install\_dependencies():  
    print("🚀 Starting Dependency Installation...")  
      
    \# 1\. Install RDKit (Binary Wheel)  
    rdkit\_path \= find\_package\_path("rdkit")  
    if rdkit\_path:  
        try:  
            import rdkit  
            print("✅ RDKit is already installed.")  
        except ImportError:  
            print("Installing RDKit...")  
            subprocess.check\_call(\[sys.executable, "-m", "pip", "install", rdkit\_path, "--no-deps", "--no-index"\])  
      
    \# 2\. Install PyTorch Extensions (Source Compilation)  
    \# Order matters\! Dependencies first.  
    pkgs \= \["mhfp", "torch\_scatter", "torch\_sparse", "torch\_cluster", "torch\_spline\_conv"\]  
      
    for pkg in pkgs:  
        try:  
            \_\_import\_\_(pkg)  
            print(f"✅ {pkg} is already installed.")  
            continue  
        except ImportError:  
            pass

        print(f"Installing {pkg} (Compiling from source)...")  
        pkg\_path \= find\_package\_path(pkg)  
        if pkg\_path:  
            \# Copy to writable temp folder to avoid Read-Only errors  
            temp\_build \= f"/kaggle/working/build\_{pkg}"  
            if os.path.exists(temp\_build): shutil.rmtree(temp\_build)  
            shutil.copytree(pkg\_path, temp\_build)  
              
            \# Handle potential double-nesting of source folders (common with .tar.gz)  
            setup\_dir \= temp\_build  
            if not os.path.exists(os.path.join(temp\_build, 'setup.py')):  
                for child in os.listdir(temp\_build):  
                    if os.path.isdir(os.path.join(temp\_build, child)):  
                        setup\_dir \= os.path.join(temp\_build, child)  
                        break  
              
            \# CRITICAL: \--no-build-isolation prevents pip from trying to download build tools from internet  
            subprocess.check\_call(\[  
                sys.executable, "-m", "pip", "install",   
                setup\_dir, "--no-deps", "--no-index", "--no-build-isolation"  
            \])

def setup\_torch\_geometric():  
    print("\\n📦 Setting up PyTorch Geometric...")  
    target\_path \= os.path.join(FINAL\_LIB\_PATH, 'torch\_geometric')  
      
    if os.path.exists(target\_path):  
        print("✅ PyG is already set up.")  
        return

    \# Move library code manually to avoid complex build errors  
    pkg\_path \= find\_package\_path("torch\_geometric")  
    if not pkg\_path: return

    temp\_extract \= '/kaggle/working/temp\_pyg\_extract'  
    if os.path.exists(temp\_extract): shutil.rmtree(temp\_extract)  
    shutil.copytree(pkg\_path, temp\_extract)

    \# Locate the inner library folder  
    inner\_lib \= None  
    for root, dirs, files in os.walk(temp\_extract):  
        if 'torch\_geometric' in dirs and os.path.exists(os.path.join(root, 'torch\_geometric', '\_\_init\_\_.py')):  
            inner\_lib \= os.path.join(root, 'torch\_geometric')  
            break  
      
    if inner\_lib:  
        shutil.copytree(inner\_lib, target\_path)  
        print(f"✅ Moved torch\_geometric to: {target\_path}")  
      
    shutil.rmtree(temp\_extract)

if \_\_name\_\_ \== "\_\_main\_\_":  
    install\_dependencies()  
    setup\_torch\_geometric()  
    \# Add to path so subsequent cells can import it  
    if FINAL\_LIB\_PATH not in sys.path:  
        sys.path.append(FINAL\_LIB\_PATH)  
    print("\\n🎉 Environment Setup Complete\!")

## **🚀 Step 3: Run Inference (Launcher Script)**

Because of name collisions between the AIMNet datasets folder and the HuggingFace datasets library, **DO NOT** run import main directly in the notebook.

Instead, create a launcher.py script that configures the system path **before** any imports happen.

**In a notebook cell:**

%%writefile /kaggle/working/launcher.py  
import sys  
import os

\# 1\. SETUP PATHS (Priority is Key)  
\# Force Python to look in our custom folders FIRST  
aimnet\_src \= '/kaggle/input/aimnet-x2d-code/AIMNet-X2D/src'  
sys.path.insert(0, aimnet\_src)

ext\_lib \= '/kaggle/working/external\_libs'  
sys.path.insert(0, ext\_lib)

\# 2\. RUN PIPELINE  
from main import main\_runner, parse\_main\_arguments  
\# ... (Your inference code here) ...

**Execute it with:**

\!python /kaggle/working/  
