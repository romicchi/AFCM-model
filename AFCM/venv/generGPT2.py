from flask import Flask, request, jsonify
from flask_cors import CORS
from decouple import config
import openai

app = Flask(__name__)
CORS(app)

# Load the OpenAI API key from the .env file
openai_api_key = config('OPENAI_API_KEY')

# Set the API key for the OpenAI client
openai.api_key = openai_api_key

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('query')

    answers = []  # To store answers

    # Generate an answer using the ChatGPT 3.5 Turbo API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ]
    )

    # Extract and store the answer
    answer = response['choices'][0]['message']['content']
    answers.append(answer)

    return jsonify({'answers': answers})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
    
    
    
