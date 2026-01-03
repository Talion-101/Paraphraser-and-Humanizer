from flask import Flask, render_template, request, jsonify
import os
from paraphraser import ParaphraserEngine, SemanticValidator
from ai_avoider import AIDetectionAvoider
import nltk

# Initialize Flask app
app = Flask(__name__)

# Initialize engines
# We do this globally so they are loaded once when the worker starts
try:
    engine = ParaphraserEngine()
    avoider = AIDetectionAvoider()
    validator = SemanticValidator()
    print("Engines initialized successfully.")
except Exception as e:
    print(f"Error initializing engines: {e}")
    # Fallback or error handling
    engine = None
    avoider = None
    validator = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/paraphrase', methods=['POST'])
def paraphrase():
    if not engine:
        return jsonify({'error': 'Server initialization failed'}), 500

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
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return "OK", 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
