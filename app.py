from flask import Flask, render_template, request, jsonify
import os
import sys
import traceback
import nltk

# Initialize Flask app
app = Flask(__name__)

# Configure NLTK data path for Render
# Render often downloads to /opt/render/nltk_data but might not have it in path by default
nltk_path = '/opt/render/nltk_data'
if os.path.exists(nltk_path):
    if nltk_path not in nltk.data.path:
        nltk.data.path.append(nltk_path)

# Import engines AFTER configuring nltk path
# We need to use try-except for imports in case of dependency issues
try:
    from paraphraser import ParaphraserEngine, SemanticValidator
    from ai_avoider import AIDetectionAvoider
except ImportError as e:
    print(f"Import Error: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    ParaphraserEngine = None
    SemanticValidator = None
    AIDetectionAvoider = None

# Initialize engines
try:
    if ParaphraserEngine:
        engine = ParaphraserEngine()
        avoider = AIDetectionAvoider()
        validator = SemanticValidator()
        print("Engines initialized successfully.", file=sys.stderr)
    else:
        engine = None
        print("Engines failed to import.", file=sys.stderr)
except Exception as e:
    print(f"Error initializing engines: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    engine = None
    avoider = None
    validator = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/paraphrase', methods=['POST'])
def paraphrase():
    if not engine:
        return jsonify({'error': 'Server initialization failed. Check server logs.'}), 500

    data = request.json
    text = data.get('text', '')
    intensity = float(data.get('intensity', 0.6))
    humanize = data.get('humanize', True)
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        # Step 1: Paraphrase
        paraphrased = engine.paraphrase(text, intensity)
        
        # Step 2: Humanize if requested
        if humanize:
            paraphrased = avoider.humanize(paraphrased, intensity)
            
        # Step 3: Validate and Improve
        paraphrased = validator.improve_paraphrase(text, paraphrased, engine)
        
        return jsonify({'result': paraphrased})
        
    except Exception as e:
        # LOG THE FULL ERROR
        print(f"Error processing request: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return "OK", 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
