from flask import Flask, render_template, request, jsonify
from generate import generate_text

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get('prompt', '').strip()
        length = int(data.get('length', 100))
        temperature = float(data.get('temperature', 0.8))
        k = int(data.get('k', 10))

        if not prompt:
            return jsonify({'error': 'Prompt cannot be empty'}), 400

        # Generate text using your model
        generated_text = generate_text(prompt, length=length, temperature=temperature, k=k)
        return jsonify({'generated_text': generated_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)