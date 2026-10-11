# Dataset Guide — DriveGuard AI 👁️

This guide explains how to download, organise, and verify the eye-state image dataset used to train the DriveGuard AI CNN model.

---

## Which Dataset to Use

We recommend **Option A** (easier and faster for beginners).

---

### Option A — Kaggle Drowsiness Detection Dataset (Recommended) ✅

**Source:** https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset  
**Size:** ~3000 images, ~150 MB  
**Classes:** Open eyes / Closed eyes (already separated into folders)

#### Steps

1. **Create a free Kaggle account** at https://www.kaggle.com (if you don't have one).

2. **Download the dataset:**
   - Go to the dataset URL above.
   - Click the **Download** button (top-right).
   - You will get a `.zip` file named something like `archive.zip`.

3. **Extract the zip file.**  
   Inside you will find folders like:
   ```
   dataset/
   ├── train/
   │   ├── Open_Eyes/
   │   └── Closed_Eyes/
   └── test/
       ├── Open_Eyes/
       └── Closed_Eyes/
   ```

4. **Copy images into our project structure:**

   | Source folder | Destination folder |
   |--------------|-------------------|
   | `train/Open_Eyes/` | `dataset/train/open/` |
   | `train/Closed_Eyes/` | `dataset/train/closed/` |
   | `test/Open_Eyes/` | `dataset/val/open/` |
   | `test/Closed_Eyes/` | `dataset/val/closed/` |

5. **Verify image counts** (run from the project root after activating venv):
   ```bash
   python -c "
   import os
   for split in ['train', 'val']:
       for cls in ['open', 'closed']:
           path = os.path.join('dataset', split, cls)
           count = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg'))])
           print(f'{split}/{cls}: {count} images')
   "
   ```
   **Target counts (minimum):**
   | Folder | Minimum |
   |--------|---------|
   | train/open | 800 |
   | train/closed | 800 |
   | val/open | 200 |
   | val/closed | 200 |

---

### Option B — MRL Eye Dataset (Larger, better quality)

**Source:** http://mrl.cs.vsb.cz/eyedataset  
**Size:** ~84,000 images (you only need a subset)  
**Classes:** Various — you will need to filter for open/closed labels.

> This dataset is more work to prepare but gives better model accuracy.  
> Only use this option if Option A is exhausted.

#### Steps for Option B
1. Download from the link above (no login required).
2. Each filename contains metadata. Images labelled `0` in the state field are **closed**, `1` are **open**.
3. Use the provided helper script (to be created in Phase 3) to filter and copy images.

---

### Option C — CEW (Closed Eyes in the Wild)

**Source:** https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html  
**Size:** ~2400 images  
**Note:** Requires manual download request.

---

## Image Requirements

Your images must meet these criteria for the CNN to train correctly:

| Property | Requirement |
|----------|-------------|
| Content | Cropped eye region only (not full face) |
| Format | `.jpg`, `.jpeg`, or `.png` |
| Minimum size | 20×20 pixels (we resize to 24×24 during training) |
| Colour | Grayscale or RGB (we convert to grayscale automatically) |
| Labels | Folder name is the label — don't mix classes |

---

## Checking Image Quality (Optional)

Run this script from the project root to check for corrupt images:

```bash
python -c "
from PIL import Image
import os

for split in ['train', 'val']:
    for cls in ['open', 'closed']:
        folder = os.path.join('dataset', split, cls)
        bad = 0
        for f in os.listdir(folder):
            if not f.endswith(('.jpg','.jpeg','.png')):
                continue
            try:
                Image.open(os.path.join(folder, f)).verify()
            except Exception:
                print(f'  Corrupt: {f}')
                bad += 1
        print(f'{split}/{cls}: {len(os.listdir(folder))} files, {bad} corrupt')
"
```

---

## Data Augmentation Note

The training script (`training/train_model.py`) applies the following augmentations automatically during training — you do **not** need to add extra images manually:

- Random horizontal flip
- Random rotation (±10°)
- Random brightness/contrast jitter
- Gaussian noise (via torchvision transforms)

These augmentations increase effective dataset size and reduce overfitting.

---

## Troubleshooting Dataset Issues

| Issue | Fix |
|-------|-----|
| Too few images (< 400 per class) | Download more from Option B or augment manually |
| Class imbalance (very different counts) | Oversample the smaller class; training script handles this |
| Images are full faces, not eyes | Crop to eye region using `training/crop_eyes.py` (created in Phase 3) |
| Corrupt images | Run the quality check script above and delete bad files |

---

*Once your dataset is in place, proceed to training:*  
```bash
python training/train_model.py
```
