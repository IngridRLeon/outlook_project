from flask import Flask, render_template, request, send_file
import pandas as pd
import pickle
from datetime import datetime
import pytz
from pickle import dump, load
from outlook_module import outlook_function  # Replace 'your_module' with the actual module name



#Define Pipeline
from transformers import pipeline
qa_pipeline = pipeline(
    "question-answering",
    model="bert-large-uncased-whole-word-masking-finetuned-squad",
    tokenizer="bert-large-uncased",
    framework="pt"  # Explicitly specify PyTorch
)


#Flask APP
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['GET', 'POST'])
def process():
    if request.method == 'POST':
        # Handle the POST request here
        user = request.form['user']
        password = request.form['password']
        start_date = datetime.strptime(request.form['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(request.form['end_date'], '%Y-%m-%d')
        
        # Call your outlook_function to get the DataFrame
        df = outlook_function(user, password, start_date.year, start_date.month, start_date.day, end_date.year, end_date.month, end_date.day, qa_pipeline)

        # Save the DataFrame to a temporary CSV file
        temp_csv_filename = 'client_vendor.csv'
        df.to_csv(temp_csv_filename, index=False)

        # Create a response to download the CSV file
        response = send_file(temp_csv_filename, as_attachment=True)
        response.headers["Content-Disposition"] = f"attachment; filename=downloaded_data.csv"
        return response

     # Handle the GET request (display the form) here
    return render_template('index.html')  # Replace with the actual HTML template

if __name__ == '__main__':
    app.run(debug=True)
