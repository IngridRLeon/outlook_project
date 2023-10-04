
from exchangelib import Credentials, Account
from datetime import datetime
import pytz
from transformers import pipeline
import torch
import numpy as np
import pandas as pd
import json
from pickle import dump


def outlook_function (email, password, start_year, start_month, start_day, end_year, end_month, end_day, qa_pipeline):
    credentials = Credentials(email, password)
    account = Account(email, credentials=credentials, autodiscover=True)

    # Define the queries
    questions = [
    "What is the vendor's name",
    "What is the client's name",
    "What is the payment rate"
    ]
    #Define Variables
    email_answers_list = []
    answers_dict = {}
    tz = pytz.timezone('America/New_York')  # timezone

    # #Define Pipeline
    # qa_pipeline = pipeline(
    #     "question-answering",
    #     model="bert-large-uncased-whole-word-masking-finetuned-squad",
    #     tokenizer="bert-large-uncased",
    #     framework="pt"  # Explicitly specify PyTorch
    # )

    # Define the start and end date with timezone information
    start_date = tz.localize(datetime(start_year, start_month, start_day, 0, 0, 0))
    end_date = tz.localize(datetime(end_year, end_month, end_day, 23, 59, 59))

    # Filter Emails based on the start and end dates
    emails = account.inbox.filter(datetime_received__range=(start_date, end_date))

    #Iterate thorugh the emails' content to feed the LLM and get the answers to the defined queries in return
    if not emails.count():
        print("No emails found.")
    else:
        # Iterate through the QuerySet and print the type of each item
        for email in emails:
            print(f"Type of item: {type(email)}")
            
            ############################################################
            # Get the Subject and Sender of the email
            #print(f"Subject: {email.subject}")
            #print(f"Sender: {email.sender.email_address}")
            
            # Get the plain text content of the email
            email_content=email.text_body

            if email_content:
                #print(f"Plain Text Content:\n{email_content}")
                answers_dict = {}

                # Process each question and store the answers
                for question in questions:
                    result = qa_pipeline(question=question, context=email_content)
                    answer = result["answer"]
                    answers_dict[question] = answer
                
                email_answers_list.append(answers_dict)
    
    # Convert the list of dictionaries into a JSON structure
    json_data = json.dumps(email_answers_list, indent=4)

    # Convert the Json Structure into a DataFrame
    df = pd.read_json(json_data)

    # Rename the columns of the dataframe
    df.columns = ["Vendor", "Client", "Payment Rate"]
    df.index=df.index+1

    return df
