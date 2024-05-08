from flask import Flask, render_template, request
import pandas as pd
from edgar import Company
import nltk
from transformers import pipeline
import edgar
import re
app = Flask(__name__)

# Install nltk and download the punkt tokenizer if not already installed
nltk.download('punkt')

# Initialize the summarizer pipeline
summarizer = pipeline('summarization', model='t5-base')

# Define the classifier model and emotions
classifier_model_name = 'bhadresh-savani/distilbert-base-uncased-emotion'
classifier_emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

# Initialize the text classification pipeline
classifier = pipeline('text-classification', model=classifier_model_name)

def find_emotional_sentences(text, emotions, threshold):
    sentences_by_emotion = {e: [] for e in emotions}
    sentences = nltk.sent_tokenize(text)
    print(f'Document has {len(text)} characters and {len(sentences)} sentences.')
    for s in sentences:
        prediction = classifier(s)
        if prediction[0]['label'] != 'neutral' and prediction[0]['score'] > threshold:
            sentences_by_emotion[prediction[0]['label']].append(s)
    for e in emotions:
        print(f'{e}: {len(sentences_by_emotion[e])} sentences')
    return sentences_by_emotion

def summarize_sentences(sentences_by_emotion, min_length, max_length):
    all_summaries = []
    for k in sentences_by_emotion.keys():
        if len(sentences_by_emotion[k]) != 0:
            text = ' '.join(sentences_by_emotion[k])
            summary = summarizer(text, min_length=min_length, max_length=max_length)
            all_summaries.append(summary[0]['summary_text'])  # Append the summary to the list
    return all_summaries

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Get the company ticker from the form
    ticker = request.form['ticker']
    
    # Define the range of years for which you want to retrieve 10-K filings
    start_year = 2018
    end_year = 2023

    # Initialize an empty list to store insights from all filings
    all_insights = []

    # Retrieve 10-K filings for the specified range of years
    filings = Company(ticker).get_filings(form="10-K")
    filings.filter(date=f"{start_year}-01-01:{end_year}-12-31")

    # Iterate over each filing for the current year
    for filing in filings:
        # Retrieve the HTML content of the filing
        html = filing.html()

        if html:
            # Extract text content from the filing
            chunked_document = ChunkedDocument(html)
        
            # Extract items (sections) from the filing based on a condition
            items = chunked_document.show_items("Item.str.contains('ITEM', case=False)", "Item")
            
            # Initialize a dictionary to store content for each item
            item_contents = {}
            
            # Iterate over each item and extract content
            for index, row in items.iterrows():
                # Store the content in reverse order to correct the concatenation
                item_contents[row["Item"]] = item_contents.get(row["Item"], "") + row["Text"]
            
            # Create a DataFrame from the item contents
            filing_df = pd.DataFrame.from_dict(item_contents, orient="index", columns=["Content"])
            filing_df.reset_index(inplace=True)
            filing_df.columns = ["Item", "Content"]
        
        # Concatenate the extracted items to the DataFrame containing all items
            all_items_df = pd.concat([all_items_df, filing_df], ignore_index=True)
            valid_items = ["Item 1", "Item 1A", "Item 7"]

            # Filter the DataFrame to keep only the rows with valid items
            filtered_df = all_items_df[all_items_df['Item'].isin(valid_items)]

            # Find emotional sentences in the text
            sentences_by_emotion = find_emotional_sentences(text, classifier_emotions, 0.7)

            # Summarize emotional sentences
            summaries = summarize_sentences(sentences_by_emotion, min_length=20, max_length=50)

            for index, row in filtered_df.iterrows():
                # Get the content from the current row
                text = row['Content']
                sentences_by_emotion = find_emotional_sentences(text, classifier_emotions, 0.7)
                summarize_sentences(sentences_by_emotion, min_length=20, max_length=50)
                all_insights.append({
                    'Year': filing.date,
                    'Item': row['Item'],
                    'Content': row['Content'],
                    'Emotional_Sentences': sentences_by_emotion,
                    'Summaries': summaries
                })

    # Render the insights on the screen
    return render_template('insights.html', insights=all_insights)

if __name__ == '__main__':
    app.run(debug=True)