# KGE-DST

## Project Overview
KGE-DST (Knowledge Graph Embedding-Driven Smart Translation) is a smart translation system enhanced by knowledge graph retrieval. The project leverages knowledge graph context augmentation and deep learning-based semantic understanding to achieve high-precision multilingual translation.

Key features include:
- Multilingual support: Chinese, English, German, Japanese, Korean, French, etc.
- Context retrieval based on knowledge graphs to enhance translation accuracy.
- Simple front-end interface with powerful back-end translation service.

## File Structure
```
project/
├── data/                # Data files (e.g., knowledge graph embeddings)
├── frontend/            # Front-end files (HTML and static assets)
│   ├── index.html       # Main front-end page
├── src/                 # Back-end service code
│   ├── main.py          # Flask main service file
│   ├── test.py          # Test scripts
├── requirements.txt     # Python dependencies
├── LICENSE              # Project license
└── README.md            # Project documentation
```

## Setup Instructions

### Prerequisites
1. **Install Python**
   Ensure you have Python 3.10 or above installed.

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/MacOS
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   Use `pip` to install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Start the Back-End Service
1. Navigate to the `src` directory and run the Flask service:
   ```bash
   cd src
   python main.py
   ```
2. Once started, the service will be available at `http://127.0.0.1:5000`.

### Start the Front-End Page
1. Open the `frontend/index.html` file in a browser.
2. The front-end page will load, allowing access to the translation interface.

### Test Translation Functionality
1. Enter text in the input box and select the source and target languages.
2. Click the "Translate" button and wait for the translation result to be displayed.

## Notes
- Ensure the back-end service is running before using the front-end page to access translation functionality.
- If the back-end and front-end run on different hosts or ports, make sure the back-end allows cross-origin requests (configured using Flask-CORS).

## Demonstration

Below is a demonstration of the project:

### Video Demonstration
If your Markdown renderer supports video embedding, you can include the `.mp4` file directly:

```html
<video controls width="600">
  <source src="/data1/yuanxujie/KGE-DST/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
```

### GIF Alternative
If your Markdown renderer does not support video embedding, consider converting your `.mp4` file to a `.gif` and embedding it:

![Project Demo](path_to_your_demo.gif)

## Author Information
Author: Xujie Yuan
Copyright © 2025