import os
import re
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

image_ext_pattern = re.compile(r"\.(jpg|jpeg|png|webp|bmp|gif)(?:\?.*)?$", re.IGNORECASE)

if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(METADATA_FILE):
    clip_vectors = np.load(EMBEDDINGS_FILE)
    with open(METADATA_FILE) as f:
        metadata = json.load(f)
else:
    clip_vectors = np.zeros((0, 512), dtype=np.float32)
    metadata = []

print(f"[INIT] Loaded {len(metadata)} metadata items with {clip_vectors.shape[0]} vectors.")

def generate_caption(image: Image.Image) -> str:
    inputs = processor(image, return_tensors="pt").to(device)
    outputs = caption_model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True)

def extract_keywords(caption: str):
    doc = nlp(caption.lower())
    return list(set(token.lemma_ for token in doc if token.is_alpha and not token.is_stop))

def jaccard_similarity(tags1, tags2):
    set1, set2 = set(tags1), set(tags2)
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

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

def crawl_images(base_url, depth=2, max_pages=10, visited=None):
    if visited is None:
        visited = set()
    if base_url in visited or len(visited) >= max_pages:
        return []

    visited.add(base_url)
    print(f"[CRAWL] {base_url}")

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    img_urls = set()
    try:
        resp = requests.get(base_url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Collect image URLs
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src")
            if src:
                full_url = urljoin(base_url, src)
                if image_ext_pattern.search(full_url):
                    img_urls.add(full_url)

        # Follow more links, including those in nav, footer, main, and aside
        link_tags = soup.select("a[href], nav a, footer a, main a, aside a")
        links = set(urljoin(base_url, a['href']) for a in link_tags if a.get('href'))

        # Filter and crawl deeper
        for link in links:
            if link not in visited and base_url.split("//")[1].split("/")[0] in link:
                img_urls.update(crawl_images(link, depth=depth - 1, max_pages=max_pages, visited=visited))
                if len(visited) >= max_pages:
                    break

        return list(img_urls)

    except Exception as e:
        print(f"[ERROR] crawl {base_url}: {e}")
        return list(img_urls)


def download_and_embed(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB").resize((512, 512))
        phash = str(imagehash.phash(img))

        # Skip if hash already exists
        if any(m.get("hash") == phash for m in metadata):
            print(f"[SKIP] Duplicate image (hash: {phash}) from {url}")
            return None, None

        filename = f"{phash}.webp"
        path = os.path.join(IMAGE_FOLDER, filename)
        img.save(path, "WEBP", quality=85)

        image_input = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image_input).cpu().numpy()[0]

        caption = generate_caption(img)
        tags = extract_keywords(caption)

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

                filtered_indices = [
                    i for i, meta in enumerate(metadata)
                    if "hash" in meta and (query_hash - imagehash.hex_to_hash(meta["hash"])) < 16
                ]

                if not filtered_indices:
                    filtered_indices = list(range(len(metadata)))

                if filtered_indices:
                    search_vectors = clip_vectors[filtered_indices]
                    if search_vectors.size > 0:
                        sims = cosine_similarity(query_embed, search_vectors)[0]

                        scored_results = []
                        for i, sim in enumerate(sims):
                            real_idx = filtered_indices[i]
                            meta = metadata[real_idx]
                            tag_score = jaccard_similarity(generated_tags, meta.get("tags", []))
                            combined_score = 0.7 * sim + 0.3 * tag_score
                            scored_results.append((combined_score, meta, sim, tag_score))

                        scored_results.sort(key=lambda x: x[0], reverse=True)

                        for score, meta, clip_score, tag_score in scored_results[:25]:
                            matches.append((meta["path"], 0, score, meta.get("caption", ""), meta.get("tags", []), meta.get("source", "")))
                    else:
                        print("[WARN] Empty search vector array.")
                else:
                    print("[WARN] No valid indices for search.")

        elif text:
            query_type = "text"
            if clip_vectors.shape[0] == 0:
                print("[WARN] clip_vectors empty, skipping text search.")
            else:
                query_embed = process_query_text(text)
                sims = cosine_similarity(query_embed, clip_vectors)[0]
                top_indices = np.argsort(sims)[-100:][::-1]

                text_lower = text.lower()
                for idx in top_indices:
                    meta = metadata[idx]
                    if (text_lower in meta.get("caption", "").lower()) or any(text_lower in t for t in meta.get("tags", [])):
                        matches.append((meta["path"], 0, sims[idx], meta.get("caption", ""), meta.get("tags", []), meta.get("source", "")))
                        if len(matches) >= 25:
                            break

        elif site_url:
            query_type = "url"
            try:
                img_urls = crawl_images(site_url)
                for url in img_urls[:15]:
                    meta, _ = download_and_embed(url)
                    if meta:
                        matches.append((meta["path"], 0, 1.0, meta["caption"], meta["tags"], meta.get("source", "")))
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
