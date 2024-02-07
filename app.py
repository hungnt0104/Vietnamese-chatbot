from flask import Flask, render_template, request, flash
from main import get_response

app = Flask(__name__)

messages = []
questions = []
answers = []

app.secret_key = "secret key"

@app.route('/')
def index():
    return render_template('index.html', messages=messages)

@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.form.get('user_message')
    if user_message:
        label = get_response(user_message)
        messages.append(user_message)
        messages.append(label)
        messages_with_index = list(enumerate(messages))
        return render_template('index.html', messages_with_index=messages_with_index)
if __name__ == '__main__':
    app.run(debug=True)