from flask import Flask, request, jsonify
from vllm import LLM, SamplingParams

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

print("Loading model with vLLM...")
llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    max_model_len=512,
    gpu_memory_utilization=0.9,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100,
)

print("Model loaded successfully!")

app = Flask(__name__)


@app.route('/compose', methods=['POST'])
def compose_sentence():
    try:
        data = request.get_json()
        
        if not data or 'pictos' not in data:
            return jsonify({"error": "Missing 'pictos' field in request"}), 400
        
        pictos = data['pictos']
        
        if not isinstance(pictos, list) or not pictos:
            return jsonify({"error": "'pictos' must be a non-empty list"}), 400
        
        root_sequence = " ".join(pictos)
        
        prompt = f"Reformula esta frase en español natural: {root_sequence}\n\nFrase reformulada:"
        print(f'Processing: {prompt}')
        
        # Generate with vLLM
        outputs = llm.generate([prompt], sampling_params)
        
        # Extract generated text
        generated_text = outputs[0].outputs[0].text.strip()
        
        return jsonify({
            "input": root_sequence,
            "output": generated_text,
            "pictos": pictos
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model": MODEL_NAME})


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Pictogram to Sentence API with vLLM",
        "endpoints": {
            "/compose": "POST - Compose sentence from pictos",
            "/health": "GET - Health check"
        }
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
