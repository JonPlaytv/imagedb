# 🔍 Smart Image Search & Tagging Web App

Ein GPU-beschleunigtes Tool zur visuellen Bildersuche, automatischen Caption-Erstellung, Tagging und Website-Bild-Scraping – alles in einer Weboberfläche.

![screenshot](https://github.com/user-attachments/assets/c97343dd-9181-4a6e-9220-6a0d826dc48f) 

## ✨ Features

- Drag & Drop oder Upload für Bildsuche
- Textsuche mit natürlichen Beschreibungen (z. B. „a dog in the snow“)
- Automatische Captions mit [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base)
- Tag-Extraktion mit [spaCy](https://spacy.io/)
- Ähnlichkeitssuche über [CLIP](https://github.com/openai/CLIP)
- GPU-Unterstützung (CUDA, wenn verfügbar)
- Website-Scraper: lädt und verarbeitet automatisch alle Bilder einer URL
- Schlagwortsuche per Klick auf Tags
- Responsive Web UI mit Lightbox, Vorschau & Galerie

![Screenshot 2025-06-24 162116](https://github.com/user-attachments/assets/09b58445-c517-4c6e-a758-706219009fa7)

## 🚀 Schnellstart

```bash
git clone https://github.com/JonPlaytv/imagedb.git
cd image-search-app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python app.py
