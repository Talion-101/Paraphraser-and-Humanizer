# Paraphraser & Humanizer - Web Deployment

This folder contains the web version of the Paraphraser & Humanizer application, ready for deployment on Render, Heroku, or any cloud platform that supports Python Flask applications.

## How to Deploy on Render

1.  **Push to GitHub/GitLab**:
    - Initialize a git repository in this folder (or the parent folder) and push it to GitHub.
    - If this is a subdirectory of a larger repo, you can configure the "Root Directory" in Render.

2.  **Create a New Web Service on Render**:
    - Go to [dashboard.render.com](https://dashboard.render.com).
    - Click **New +** -> **Web Service**.
    - Connect your repository.

3.  **Configuration**:
    - **Name**: Give it a name (e.g., `paraphraser-app`).
    - **Runtime**: `Python 3`.
    - **Build Command**: `pip install -r requirements.txt`. (Render usually detects this automatically).
    - **Start Command**: `gunicorn app:app`. (Render will read this from the `Procfile` automatically).
    - **Root Directory**: If you uploaded the entire project, set this to `web_deploy`. If you uploaded just this folder as the root of the repo, leave it blank.

4.  **Environment Variables**:
    - No special environment variables are needed for the basic functionality.

## Local Testing

To run this web app locally before deploying:

1.  Open a terminal in this folder.
2.  Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the app:
    ```bash
    python app.py
    ```
4.  Open your browser to `http://localhost:5000`.

## Files Structure

- `app.py`: The Flask web server.
- `paraphraser.py`: Core paraphrasing logic (ported from desktop app).
- `ai_avoider.py`: AI detection avoidance logic.
- `templates/index.html`: The dark-themed web interface.
- `requirements.txt`: Python dependencies.
- `Procfile`: Command for Render to start the app.
