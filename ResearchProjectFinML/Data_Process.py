import nltk
import pandas as pd
from transformers import pipeline


summarizer = pipeline('summarization', model='t5-base')

classifier_model_name = 'bhadresh-savani/distilbert-base-uncased-emotion'
classifier_emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

classifier = pipeline('text-classification', model=classifier_model_name)

def find_emotional_sentences(text, emotions, threshold):
    sentences_by_emotion = {}
    for e in emotions:
        sentences_by_emotion[e]=[]
    sentences = nltk.sent_tokenize(text)
    print(f'Document has {len(text)} characters and {len(sentences)} sentences.')
    for s in sentences:
        prediction = classifier(s)
        if (prediction[0]['label']!='neutral' and prediction[0]['score']>threshold):
            #print (f'Sentence #{sentences.index(s)}: {prediction} {s}')
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

# Initialize dictionaries to store aggregated emotional sentences and their summaries
all_sentences_by_emotion = {emotion: [] for emotion in classifier_emotions}
all_summaries = []

# Iterate over each row in the filtered DataFrame
for index, row in filtered_df.iterrows():
    # Get the content from the current row
    text = row['Content']
    
    # Find emotional sentences in the text
    sentences_by_emotion = find_emotional_sentences(text, classifier_emotions, 0.7)
    
    # Summarize emotional sentences
    summaries = summarize_sentences(sentences_by_emotion, min_length=20, max_length=50)
    
    # Aggregate emotional sentences and summaries
    for emotion, sentences in sentences_by_emotion.items():
        all_sentences_by_emotion[emotion].extend(sentences)
    
    # Aggregate summaries
    all_summaries.extend(summaries)

# Generate final insight based on aggregated data
final_insight = {
    "emotional_sentences": all_sentences_by_emotion,
    "summaries": all_summaries
}

# Print or return the final insight
print(final_insight)