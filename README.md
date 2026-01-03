# Paraphraser & Humanizer - Web Version for Render

This folder contains a web version of the Paraphraser & Humanizer application that can be deployed to Render.com.

## Features

- Web-based interface for text paraphrasing
- AI detection avoidance
- REST API endpoints for programmatic access
- Flask-based backend with the same core algorithms as the desktop version

## Deployment to Render

### Prerequisites

1. A Render.com account
2. GitHub repository with this code
3. Python 3.7+ environment

### Deployment Steps

1. **Connect your GitHub repository** to Render
2. **Create a new Web Service** and select this repository
3. **Configure the service**:
   - Name: `paraphraser-web`
   - Region: Choose the closest to your users
   - Branch: `main` (or your deployment branch)
   - Root Directory: `render` (if this folder is in a subdirectory)
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn -b 0.0.0.0:$PORT app:app`
   - Environment: Python 3

4. **Set environment variables** (optional):
   - `FLASK_ENV=production`
   - `PYTHONUNBUFFERED=1`

5. **Deploy** the service

### Post-Deployment

- The application will be available at your Render service URL
- The first deployment may take a few minutes as dependencies are installed
- NLTK data will be downloaded automatically on first run

## API Endpoints

### POST /api/paraphrase

Paraphrase text with optional humanization.

**Request Body:**
```json
{
    "text": "Your text to paraphrase",
    "intensity": 0.6,  // 0.0 to 1.0
    "humanize": true   // boolean
}
```

**Response:**
```json
{
    "original": "Your original text",
    "paraphrased": "Paraphrased result",
    "success": true
}
```

### POST /api/validate

Validate paraphrased text for semantic similarity.

**Request Body:**
```json
{
    "original": "Original text",
    "paraphrased": "Paraphrased text"
}
```

**Response:**
```json
{
    "validation": {
        "similarity_score": 85.5,
        "semantic_match": true,
        "is_humanized": true,
        "quality_status": "GOOD",
        "recommendations": []
    },
    "success": true
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
    "status": "healthy"
}
```

## Local Development

To run locally:

```bash
cd render
pip install -r requirements.txt
python app.py
```

The application will be available at `http://localhost:5000`

## File Structure

```
render/
├── app.py                  # Flask application
├── paraphraser.py          # Core paraphrasing engine
├── ai_avoider.py           # AI detection avoidance
├── requirements.txt        # Python dependencies
├── render.yaml             # Render deployment config
├── templates/
│   └── index.html          # Web interface
└── README.md               # This file
```

## Notes

- The web version uses the same core algorithms as the desktop version
- All processing is done server-side
- The application is stateless and can be scaled horizontally
- NLTK data is downloaded automatically on first run