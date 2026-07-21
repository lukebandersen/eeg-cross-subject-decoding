"""
Extract ViT-H-14 (laion2b_s32b_b79k) image features for Alljoined1,
matching the THINGS pipeline in eegdatasets.py EXACTLY:
  - OpenCLIP ViT-H-14, pretrained='laion2b_s32b_b79k', precision fp32
  - same preprocess transform from create_model_and_transforms
  - RGB convert, batch of 20, encode_image under no_grad
  - RAW (unnormalized) features, just like _encode_images (loss normalizes later)

Output: a dict saved as alljoined_ViT-H-14_features.pt
  {
    'coco_ids': LongTensor (M,),          # image id order
    'img_features': FloatTensor (M, 1024) # raw ViT-H-14 embeddings, same order
  }

The coco_ids come from the loader's train/test coco_id lists, so features
align by id. This does NOT build text features; the gate test uses image
retrieval only. (THINGS' "This picture is {class}" text convention does not
transfer to COCO; revisit if the text branch is needed later.)

Images are fetched by coco_id from the COCO image server, so only the
~960 unique images you actually use get downloaded.
"""

import io
import os
import time
import urllib.request

import numpy as np
import torch
from PIL import Image

# reuse the loader to get the exact coco_ids we need features for
import alljoined_loader as L

OUT_PATH = r"C:/Users/mwolff3/Desktop/alljoined1/alljoined_ViT-H-14_features.pt"
IMG_CACHE = r"C:/Users/mwolff3/Desktop/alljoined1/coco_images"
CLIP_MODEL_TYPE = "ViT-H-14"
CLIP_PRETRAINED = "laion2b_s32b_b79k"
BATCH = 20

# COCO id -> image URL. NSD test images are all from COCO train2017/val2017.
# The COCO servers serve by zero-padded 12-digit id. We try train2017 first,
# then val2017, since NSD draws from both.
COCO_URL_TEMPLATES = [
    "http://images.cocodataset.org/train2017/{:012d}.jpg",
    "http://images.cocodataset.org/val2017/{:012d}.jpg",
]


def _load_clip():
    import open_clip
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cache_dir = os.environ.get("OPEN_CLIP_CACHE_DIR", None)
    model, preprocess_train, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL_TYPE, pretrained=CLIP_PRETRAINED,
        precision="fp32", device=device, cache_dir=cache_dir)
    model.eval()
    print(f"OpenCLIP {CLIP_MODEL_TYPE} ({CLIP_PRETRAINED}) loaded on {device}")
    return model, preprocess_train, device


def _fetch_image(coco_id):
    """Return a PIL RGB image for a coco_id, caching to disk. None if unreachable."""
    os.makedirs(IMG_CACHE, exist_ok=True)
    local = os.path.join(IMG_CACHE, f"{coco_id:012d}.jpg")
    if os.path.exists(local):
        try:
            return Image.open(local).convert("RGB")
        except Exception:
            os.remove(local)  # corrupt cache entry, refetch
    for tmpl in COCO_URL_TEMPLATES:
        url = tmpl.format(coco_id)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=20).read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            with open(local, "wb") as f:
                f.write(data)
            return img
        except Exception:
            continue
    return None


def extract_for_ids(coco_ids, model, preprocess, device):
    """Fetch + encode a list of coco_ids. Returns (kept_ids, features, missing)."""
    kept_ids, tensors, missing = [], [], []
    pil_batch, batch_ids = [], []

    def flush():
        if not pil_batch:
            return
        imgs = torch.stack([preprocess(im) for im in pil_batch]).to(device)
        with torch.no_grad():
            feats = model.encode_image(imgs)   # RAW, unnormalized (matches THINGS)
        tensors.append(feats.cpu())
        kept_ids.extend(batch_ids)
        pil_batch.clear()
        batch_ids.clear()

    for n, cid in enumerate(coco_ids, 1):
        img = _fetch_image(int(cid))
        if img is None:
            missing.append(int(cid))
            continue
        pil_batch.append(img)
        batch_ids.append(int(cid))
        if len(pil_batch) == BATCH:
            flush()
        if n % 100 == 0:
            print(f"  {n}/{len(coco_ids)} processed, {len(missing)} missing so far")
    flush()

    feats = torch.cat(tensors, dim=0) if tensors else torch.empty(0)
    return kept_ids, feats, missing


def main(subject_id=6):
    # get the exact coco_ids used by train and test for this subject
    data = L.load_subject(subject_id=subject_id, verbose=False)
    all_ids = np.unique(np.concatenate(
        [data["train_coco_ids"], data["test_coco_ids"]]))
    print(f"unique coco_ids needing features: {len(all_ids)}")

    model, preprocess, device = _load_clip()
    t0 = time.time()
    kept_ids, feats, missing = extract_for_ids(all_ids, model, preprocess, device)
    print(f"done in {time.time()-t0:.0f}s")
    print(f"  features: {feats.shape}   kept {len(kept_ids)}   missing {len(missing)}")
    if missing:
        print(f"  MISSING coco_ids (first 20): {missing[:20]}")
        print("  If many are missing, the COCO split guess is wrong; tell me and")
        print("  we switch to the NSD 73k_id route via the OSF stimulus file.")

    torch.save({
        "coco_ids": torch.tensor(kept_ids, dtype=torch.long),
        "img_features": feats.float(),
    }, OUT_PATH)
    print(f"saved -> {OUT_PATH}")
    print(f"embedding width: {feats.shape[-1] if feats.numel() else 'N/A'} (expect 1024)")


if __name__ == "__main__":
    main(subject_id=6)
