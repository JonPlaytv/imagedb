import os
import json
import numpy as np
from PIL import Image
import imagehash
import clip
import torch
import spacy
from transformers import BlipProcessor, BlipForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # <--- neu

IMAGE_FOLDER = r"E:\imagedb\static\images"
EMBEDDINGS_FILE = "clip_embeddings.npy"
METADATA_FILE = "metadata.json"
BATCH_SIZE = 16

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
nlp = spacy.load("en_core_web_sm")

def generate_caption(image: Image.Image) -> str:
    inputs = processor(image, return_tensors="pt").to(device)
    outputs = caption_model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True)

def extract_keywords(caption: str):
    doc = nlp(caption.lower())
    return list(set(token.lemma_ for token in doc if token.is_alpha and not token.is_stop))

def load_and_prepare_image(filename):
    path = os.path.join(IMAGE_FOLDER, filename)
    try:
        img = Image.open(path).convert("RGB").resize((512, 512))
        phash = str(imagehash.phash(img))
        return filename, img, phash
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return filename, None, None

def save_image(img, filename):
    path = os.path.join(IMAGE_FOLDER, filename)
    img.save(path, "WEBP", quality=85)

def load_existing_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return []

def load_existing_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        return np.load(EMBEDDINGS_FILE)
    return np.empty((0, 512))

def main():
    metadata = load_existing_metadata()
    embeddings = load_existing_embeddings()

    existing_files = set(item["path"].split("/")[-1] for item in metadata)
    existing_hashes = set(item.get("hash") for item in metadata if "hash" in item)

    all_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"))]
    to_process = [f for f in all_files if f not in existing_files]

    # Load images + compute hashes concurrently (IO bound)
    with ThreadPoolExecutor(max_workers=8) as executor:
        loaded = list(tqdm(executor.map(load_and_prepare_image, to_process), total=len(to_process), desc="Loading images"))

    # Filter out errors and existing hashes
    filtered = [(fname, img, phash) for fname, img, phash in loaded if img is not None and phash not in existing_hashes]

    if not filtered:
        print("No new images to process.")
        return

    # Batch CLIP preprocessing
    images = [preprocess(img).unsqueeze(0) for _, img, _ in filtered]
    images_tensor = torch.cat(images).to(device)

    with torch.no_grad():
        embeddings_batch = model.encode_image(images_tensor).cpu().numpy()

    new_metadata = []
    new_embeddings = []

    # Caption generation and metadata building with progress bar
    for i, (fname, img, phash) in enumerate(tqdm(filtered, desc="Processing images")):
        save_image(img, fname)  # overwrite with webp version if desired

        caption = generate_caption(img)
        tags = extract_keywords(caption)

        meta = {
            "path": os.path.join("static/images", fname).replace("\\", "/"),
            "caption": caption,
            "tags": tags,
            "hash": phash,
            "source_url": None
        }

        new_metadata.append(meta)
        new_embeddings.append(embeddings_batch[i])

    # Save updated embeddings & metadata
    all_embeddings = np.vstack([embeddings, new_embeddings])
    np.save(EMBEDDINGS_FILE, all_embeddings)

    all_metadata = metadata + new_metadata
    with open(METADATA_FILE, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"✅ Processed {len(new_metadata)} new images.")

if __name__ == "__main__":
    main()
