# Data Version Control (DVC) with Google Drive Guide

## Overview
This repository uses **DVC (Data Version Control)** integrated with **Google Drive (`dvc-gdrive`)** to:
1. **Version Large Datasets & Caches**: Track multi-gigabyte files (e.g., 2.7 GB Annotation Run 2 `embeddings.npy`, 340 MB `annotations.csv`, 263 MB feature caches) without hitting GitHub's 100 MB hard limit.
2. **Offload Heavy Git Blobs**: Keep the Git repository lightweight, fast, and clean. Git tracks tiny text pointer files (`.dvc`), while Google Drive securely stores the underlying binaries.
3. **Zero-Copy Disk Efficiency**: Configured with `cache.type = reflink,hardlink,symlink,copy`. On our Linux ext4 filesystem, DVC creates hardlinks/symlinks rather than duplicating files on disk, saving tens of gigabytes of NVMe storage.

---

## 1. Quick Environment Setup

A dedicated runner is available at [`code/current/mobilenet/bash_scripts/dvc.sh`](../bash_scripts/dvc.sh). It automatically invokes the `pt-env` DVC binary with per-user site cache directories (`DVC_SITE_CACHE_DIR=/tmp/dvc-${USER}`).

To use `dvc` directly from anywhere in your shell, add this alias to your `~/.bashrc`:
```bash
echo "alias dvc='code/current/mobilenet/bash_scripts/dvc.sh'" >> ~/.bashrc
source ~/.bashrc
```

Verify the setup:
```bash
./code/current/mobilenet/bash_scripts/dvc.sh version
```
*(Confirms `gdrive (pydrive2)` support and `hardlink, symlink` cache types).*

---

## 2. Google Drive Remote Configuration

### Step 2.1: Create a Folder on Google Drive
1. Open Google Drive in your web browser: [drive.google.com](https://drive.google.com).
2. Create a new folder (e.g. `vmmr-dvc-storage` or `econ-dvc-storage`).
3. Open the folder. Look at the browser URL bar:
   ```text
   https://drive.google.com/drive/folders/1ABC123xyz_YourFolderIDHere456
   ```
   The string `1ABC123xyz_YourFolderIDHere456` is your **Google Drive Folder ID**.

### Step 2.2: Set the Remote URL in DVC
Run the following command, replacing `<FOLDER_ID>` with your actual folder ID:
```bash
./code/current/mobilenet/bash_scripts/dvc.sh remote modify gdrive-storage url "gdrive://<FOLDER_ID>"
```

### Step 2.3: Service Account Authentication (Current & Recommended)
The repository is configured to authenticate headlessly using a Google Cloud Service Account (`research-pc@vmmr-dvc.iam.gserviceaccount.com`). This completely eliminates browser OAuth logins, redirect loops, and port 8090 forwarding.

* **Repository Configuration**:
  `.dvc/config` has:
  ```ini
  ['remote "gdrive-storage"']
      url = gdrive://1jmsSPZKa0o1LpOW8mgZFg1kghPBolerc
      gdrive_use_service_account = true
  ```
* **Local Credentials (Git-Ignored)**:
  `.dvc/config.local` points to the private JSON key:
  ```ini
  ['remote "gdrive-storage"']
      gdrive_service_account_json_file_path = /home/researchadmin/Econ-n/repo-clone/vmmr/.dvc/vmmr-dvc-bb9468f21bc7.json
  ```
  *(Note: All `*.json` files inside `.dvc/` are git-ignored to prevent credential leakage).*

* **Sharing Requirement**:
  The Google Drive folder (`1jmsSPZKa0o1LpOW8mgZFg1kghPBolerc`) is shared with `research-pc@vmmr-dvc.iam.gserviceaccount.com` as **Editor**.

* **Daily Usage**:
  Push and pull operate completely headlessly:
  ```bash
  ./code/current/mobilenet/bash_scripts/dvc.sh status -r gdrive-storage
  ./code/current/mobilenet/bash_scripts/dvc.sh push
  ./code/current/mobilenet/bash_scripts/dvc.sh pull
  ```

---

### Step 2.4: Legacy Interactive User OAuth (Alternative / Fallback)
If you ever need to use personal user OAuth instead of a Service Account:
1. Unset the service account flag:
   ```bash
   ./code/current/mobilenet/bash_scripts/dvc.sh remote modify --unset gdrive-storage gdrive_use_service_account
   ```
2. Run via [`code/current/mobilenet/bash_scripts/dvc_ephemeral.sh`](../bash_scripts/dvc_ephemeral.sh):
   ```bash
   export DVC_GDRIVE_CLIENT_ID="your_client_id.apps.googleusercontent.com"
   export DVC_GDRIVE_CLIENT_SECRET="your_client_secret"
   ./code/current/mobilenet/bash_scripts/dvc_ephemeral.sh push
   ```


---

## 3. Recommended Files & Folders to Track

| Target Path | Size | Reason for DVC Tracking |
| :--- | :--- | :--- |
| `code/current/mobilenet/annotation_runs/run2/` | **3.3 GB** | 1.12M image Qwen annotations, 2.7 GB `embeddings.npy`, 340 MB `annotations.csv`, color cache |
| `code/current/mobilenet/color_distillation/cache/` | **304 MB** | Pre-extracted MobileNetV2 L16 features (`features_L16_train.npy`, `features_L16_val.npy`) |
| `code/current/mobilenet/color_distillation/output/mobilenet_v2_multitask.onnx` | **20.6 MB** | Dual-head ONNX model artifact (can be migrated from git to dvc) |
| Future teacher / student models (`*.pt`, `*.onnx`) | **20 - 200 MB** | Future multi-head teacher and student distilled checkpoints |

---

## 4. Daily Workflow Commands

### Adding New Datasets or Caches to DVC
```bash
# 1. Track the directory with DVC
./code/current/mobilenet/bash_scripts/dvc.sh add code/current/mobilenet/annotation_runs/run2

# 2. Track the feature cache
./code/current/mobilenet/bash_scripts/dvc.sh add code/current/mobilenet/color_distillation/cache

# 3. Commit the lightweight .dvc pointers to Git (autostage is enabled)
git commit -m "Track Annotation Run 2 annotations and color distillation cache in DVC"

# 4. Push large files to Google Drive
./code/current/mobilenet/bash_scripts/dvc.sh push

# 5. Push code and .dvc pointers to GitHub
git push origin main
```

### Migrating an Existing Git-Tracked Model to DVC
If a file was previously committed to Git (e.g. `mobilenet_v2_multitask.onnx`):
```bash
# 1. Untrack from Git index (keeps file on disk)
git rm --cached code/current/mobilenet/color_distillation/output/mobilenet_v2_multitask.onnx

# 2. Add to DVC
./code/current/mobilenet/bash_scripts/dvc.sh add code/current/mobilenet/color_distillation/output/mobilenet_v2_multitask.onnx

# 3. Commit to Git
git commit -m "Migrate multitask ONNX model from Git to DVC"

# 4. Push to Google Drive
./code/current/mobilenet/bash_scripts/dvc.sh push
```

### Pulling Data on Another Machine
When cloning or pulling the repo on another server:
```bash
git pull origin main
./code/current/mobilenet/bash_scripts/dvc.sh pull
```
*(DVC downloads only the files needed for the current git checkout).*

---

## 5. DVC Repository Configuration Details
The `.dvc/config` file is committed to Git:
```ini
[cache]
    type = "reflink,hardlink,symlink,copy"
[core]
    analytics = false
    remote = gdrive-storage
    autostage = true
['remote "gdrive-storage"']
    url = gdrive://<FOLDER_ID>
```
- `autostage = true`: When you run `dvc add <path>`, DVC automatically stages `<path>.dvc` and `.gitignore` in Git.
- `cache.type = reflink,hardlink,symlink,copy`: Avoids duplicate storage on disk.
- `.dvcignore`: Excludes unwanted temp files from DVC hashing.
