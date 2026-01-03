# Flask application for Paraphraser and Humanizer web service
from flask import Flask, request, jsonify, render_template
from paraphraser import ParaphraserEngine, SemanticValidator
from ai_avoider import AIDetectionAvoider
import os

app = Flask(__name__)

# Initialize engines
engine = ParaphraserEngine()
avoider = AIDetectionAvoider()
validator = SemanticValidator()

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/api/paraphrase', methods=['POST'])
def paraphrase():
    """API endpoint for paraphrasing text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        intensity = data.get('intensity', 0.6)
        humanize = data.get('humanize', True)
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        # Paraphrase the text
        paraphrased = engine.paraphrase(text, intensity)
        
        # Humanize if requested
        if humanize:
            paraphrased = avoider.humanize(paraphrased, intensity)
        
        # Validate and improve
        paraphrased = validator.improve_paraphrase(text, paraphrased, engine)
        
        return jsonify({
            'original': text,
            'paraphrased': paraphrased,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate', methods=['POST'])
def validate():
    """API endpoint for validating paraphrased text"""
    try:
        data = request.get_json()
        original_text = data.get('original', '')
        paraphrased_text = data.get('paraphrased', '')
        
        if not original_text or not paraphrased_text:
            return jsonify({'error': 'Both original and paraphrased text required'}), 400
        
        validation = validator.calculate_semantic_similarity(original_text, paraphrased_text)
        
        return jsonify({
            'validation': validation,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)