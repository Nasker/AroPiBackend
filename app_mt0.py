import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# MT0 is better for multilingual tasks including Catalan
# Options: "bigscience/mt0-small" (300M), "bigscience/mt0-base" (580M), "bigscience/mt0-large" (1.2B)
# For better quality, use mt0-large (requires more RAM/VRAM)
MODEL_NAME = "bigscience/mt0-large"  # Best quality for Catalan and Spanish

# Force CPU for now (MX230 GPU not supported - compute capability 6.1 < 7.0)
# On a machine with modern GPU (RTX 20xx+), change to: torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

print(f"Loading {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
)
model = model.to(DEVICE)
model.eval()

print("Model loaded successfully!")

app = Flask(__name__)


@app.route('/compose', methods=['POST'])
def compose_sentence():
    """
    Endpoint to compose a natural sentence from pictograms.
    Expected JSON: {
        "pictos": ["yo", "querer", "agua"],
        "language": "es"  # Optional: "es" (Spanish), "ca" (Catalan), "en" (English)
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'pictos' not in data:
            return jsonify({"error": "Missing 'pictos' field in request"}), 400
        
        pictos = data['pictos']
        language = data.get('language', 'es')  # Default to Spanish
        
        if not isinstance(pictos, list) or not pictos:
            return jsonify({"error": "'pictos' must be a non-empty list"}), 400
        
        # Join pictos → "yo querer agua"
        root_sequence = " ".join(pictos)
        
        # Create language-specific prompts with better instructions
        prompts = {
            'es': f"Convierte estas palabras en una frase completa y natural en español: {root_sequence}\n\nFrase:",
            'ca': f"Converteix aquestes paraules en una frase completa i natural en català: {root_sequence}\n\nFrase:",
            'en': f"Convert these words into a complete and natural English sentence: {root_sequence}\n\nSentence:"
        }
        
        prompt = prompts.get(language, prompts['es'])
        print(f'Processing [{language}]: {prompt}')
        
        # Tokenize and generate with MT0
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                min_new_tokens=5,  # Ensure minimum output length
                temperature=0.8,  # Slightly higher for more natural variation
                top_p=0.95,  # Broader sampling for better quality
                top_k=50,  # Add top-k sampling
                do_sample=True,
                num_beams=4,  # Beam search for better quality (slower but better)
                early_stopping=True,
                repetition_penalty=1.2,  # Reduce repetition
                no_repeat_ngram_size=3,  # Prevent 3-gram repetition
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({
            "input": root_sequence,
            "output": generated_text,
            "pictos": pictos,
            "language": language
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model": MODEL_NAME,
        "device": str(DEVICE)
    })


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Pictogram to Sentence API with MT0",
        "model": MODEL_NAME,
        "supported_languages": ["es (Spanish)", "ca (Catalan)", "en (English)"],
        "endpoints": {
            "/compose": "POST - Compose sentence from pictos",
            "/health": "GET - Health check"
        },
        "example": {
            "pictos": ["jo", "voler", "aigua"],
            "language": "ca"
        }
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
