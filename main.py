import os
import json
import numpy as np
import clip
import torch
import imagehash
import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from flask import Flask, request, render_template
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BlipProcessor, BlipForConditionalGeneration
import spacy
from urllib.parse import urljoin

app = Flask(__name__)

IMAGE_FOLDER = "static/images"
EMBEDDINGS_FILE = "clip_embeddings.npy"
METADATA_FILE = "metadata.json"

os.makedirs(IMAGE_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

nlp = spacy.load("en_core_web_sm")

# Lade bestehende Embeddings und Metadaten, falls vorhanden
if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(METADATA_FILE):
    clip_vectors = np.load(EMBEDDINGS_FILE)
    with open(METADATA_FILE) as f:
        metadata = json.load(f)
else:
    clip_vectors = np.zeros((0, 512), dtype=np.float32)
    metadata = []

def generate_caption(image: Image.Image) -> str:
    inputs = processor(image, return_tensors="pt").to(device)
    outputs = caption_model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True)

def extract_keywords(caption: str):
    doc = nlp(caption.lower())
    return list(set(token.lemma_ for token in doc if token.is_alpha and not token.is_stop))

def process_query_image(image):
    query_hash = imagehash.phash(image)
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        query_embedding = model.encode_image(image_input).cpu().numpy()
    return query_hash, query_embedding

def process_query_text(text):
    text_input = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_input).cpu().numpy()
    return text_embedding

def get_images_from_website(base_url):
    print(f"[INFO] Scraping URL: {base_url}")
    try:
        resp = requests.get(base_url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        img_urls = set()
        for img in soup.find_all("img"):
            src = img.get("src")
            if src:
                full = urljoin(base_url, src)
                if full.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")):
                    img_urls.add(full)
        print(f"[INFO] Found {len(img_urls)} images")
        return list(img_urls)
    except Exception as e:
        print(f"[ERROR] Failed to scrape {base_url}: {e}")
        return []

def download_and_embed(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img = img.resize((512, 512))
        filename = f"web_{len(metadata)}.webp"
        path = os.path.join(IMAGE_FOLDER, filename)
        img.save(path, "WEBP", quality=85)

        image_input = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image_input).cpu().numpy()[0]

        caption = generate_caption(img)
        tags = extract_keywords(caption)
        phash = str(imagehash.phash(img))

        meta = {
            "path": f"{IMAGE_FOLDER}/{filename}".replace("\\", "/"),
            "caption": caption,
            "tags": tags,
            "hash": phash,
            "source": url
        }

        metadata.append(meta)

        global clip_vectors
        clip_vectors = np.vstack([clip_vectors, embedding[None]])

        return meta, embedding
    except Exception as e:
        print(f"[SKIP] {url}: {e}")
        return None, None

@app.route("/", methods=["GET", "POST"])
def index():
    matches = []
    generated_caption = None
    generated_tags = []
    query_type = None

    if request.method == "POST":
        file = request.files.get("query")
        text = request.form.get("text_query", "").strip()
        site_url = request.form.get("site_query", "").strip()

        if file and file.filename != "":
            query_type = "image"
            try:
                query_img = Image.open(file.stream).convert("RGB")
            except Exception as e:
                print(f"[ERROR] Failed to open uploaded image: {e}")
                query_img = None

            if query_img:
                generated_caption = generate_caption(query_img)
                generated_tags = extract_keywords(generated_caption)
                query_hash, query_embed = process_query_image(query_img)

                # Filter metadata by perceptual hash distance
                filtered_indices = [
                    i for i, meta in enumerate(metadata)
                    if (query_hash - imagehash.hex_to_hash(meta["hash"])) < 16
                ]
                if not filtered_indices:
                    filtered_indices = list(range(len(metadata)))

                search_vectors = clip_vectors[filtered_indices]
                sims = cosine_similarity(query_embed, search_vectors)[0]
                top_indices = np.argsort(sims)[-25:][::-1]

                for idx in top_indices:
                    real_idx = filtered_indices[idx]
                    meta = metadata[real_idx]
                    matches.append((meta["path"], 0, sims[idx], meta.get("caption", ""), meta.get("tags", [])))

        elif text:
            query_type = "text"
            query_embed = process_query_text(text)
            sims = cosine_similarity(query_embed, clip_vectors)[0]
            top_indices = np.argsort(sims)[-100:][::-1]

            text_lower = text.lower()
            for idx in top_indices:
                meta = metadata[idx]
                if (text_lower in meta.get("caption", "").lower()) or any(text_lower in t for t in meta.get("tags", [])):
                    matches.append((meta["path"], 0, sims[idx], meta.get("caption", ""), meta.get("tags", [])))
                    if len(matches) >= 25:
                        break

        elif site_url:
            query_type = "url"
            try:
                img_urls = get_images_from_website(site_url)
                for url in img_urls[:15]:
                    meta, _ = download_and_embed(url)
                    if meta:
                        matches.append((meta["path"], 0, 1.0, meta["caption"], meta["tags"]))
                # Save DB after new downloads
                np.save(EMBEDDINGS_FILE, clip_vectors)
                with open(METADATA_FILE, "w") as f:
                    json.dump(metadata, f, indent=2)
            except Exception as e:
                print(f"[ERROR] Processing site URL {site_url}: {e}")

    return render_template(
        "index.html",
        matches=matches,
        generated_caption=generated_caption,
        generated_tags=generated_tags,
        query_type=query_type,
    )

if __name__ == "__main__":
    app.run(debug=True)
