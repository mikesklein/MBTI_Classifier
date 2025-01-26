from ollama import chat
from ollama import ChatResponse
import re
import pandas as pd
import sys

MODEL = 'llama3.1:8b'
INPUT_FILE = 'data.csv'
RUN = 3
SAMPLE_SIZE = 400
OUTPUT_FILE_NAME = f"results_{MODEL}_{RUN}"

"""
A script to classify tweets based on the Myers-Briggs Personality Type scale. You need Ollama installed locally to run this script.
"""
def generate(prompt, data):
    #
    """
    Give the model a task and content, generate a response from the model and return the result.
    """
    response: ChatResponse = chat(model=MODEL, stream=False, messages=[
      {
        'role': 'user',
        'content': f"{prompt}{data}"
      },
    ])
    return strip_think_tags(response.message.content)

def strip_think_tags(response_text):
    """
    Strips everything between <think> and </think> tags from the response.
    """
    stripped = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
    return stripped

def load_csv(file_path):
    """
    Load a CSV file and return a list of dictionaries.
    """
    df = pd.read_csv(file_path)
    return df

def classify(data):
    """
    Classify the data using the model.
    """
    # use the generate function to classify the data into the MTBI categories
    mbti_regex = r'\b(?:INTJ|INTP|ENTJ|ENTP|INFJ|INFP|ENFJ|ENFP|ISTJ|ISFJ|ESTJ|ESFJ|ISTP|ISFP|ESTP|ESFP)\b'

    prompt = f"Predict what this person’s personality type using the Myers-Briggs Personality Type scale.  In your response, provide only the acronym of the personality type you think the person is. Again, only provide a four letter response based on Myers-Briggs. Here are their last 50 Tweets:"
    response = generate(prompt, data).strip()
    match = re.search(mbti_regex, response)
    return match.group(0) if match else "UNKNOWN"


def create_classification_csv(input_csv, output_csv):
    """
    Classifies data from the input CSV, compares predictions with known types,
    and saves results to a new CSV file.

    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to save the output CSV file.
    """
    # df = pd.read_csv(input_csv)
    df = pd.read_csv(input_csv)
    sampled_df = df.sample(n=SAMPLE_SIZE)
    # sampled_df = df.sample(n=SAMPLE_SIZE, random_state=42)

    predictions = []
    for index, row in sampled_df.iterrows():
        # print(row['posts'])


        predicted_type = classify(row['posts'])  # classify function assumed to exist
        print(f"Index: {index}, Type: {row['type']}, Predicted: {predicted_type}")
        is_correct = row['type'] == predicted_type

        EvI = row['type'][0] == predicted_type[0]
        SvN = row['type'][1] == predicted_type[1]
        TvF = row['type'][2] == predicted_type[2]
        JvP = row['type'][3] == predicted_type[3]

        predictions.append({
            'Index': index,
            'Type': row['type'],
            'Prediction': predicted_type,
            'Correct': is_correct,
            'EvI': EvI,
            'SvN': SvN,
            'TvF': TvF,
            'JvP': JvP,
            'Model': MODEL,
            'Posts': row['posts']
        })

    # Create a new DataFrame from the predictions and save to a CSV
    results_df = pd.DataFrame(predictions)
    results_df.to_csv(output_csv, index=False)

def calculate_accuracy(input_csv):
    """
    Calculate the accuracy of the model on the input CSV data.

    Args:
        input_csv (str): Path to the input CSV file.

    Returns:
        float: The accuracy of the model as a percentage.
    """
    df = pd.read_csv(input_csv)
    correct_predictions = df['Correct'].sum()
    correct_EvI = df['EvI'].sum()
    correct_SvN = df['SvN'].sum()
    correct_TvF = df['TvF'].sum()
    correct_JvP = df['JvP'].sum()

    total_predictions = len(df)
    total_accuracy = correct_predictions / total_predictions * 100
    EvI_accuracy = correct_EvI / total_predictions * 100
    SvN_accuracy = correct_SvN / total_predictions * 100
    TvF_accuracy = correct_TvF / total_predictions * 100
    JvP_accuracy = correct_JvP / total_predictions * 100

    print(f"Total Accuracy: {total_accuracy:.2f}%")
    print(f"EvI Accuracy: {EvI_accuracy:.2f}%")
    print(f"SvN Accuracy: {SvN_accuracy:.2f}%")
    print(f"TvF Accuracy: {TvF_accuracy:.2f}%")
    print(f"JvP Accuracy: {JvP_accuracy:.2f}%")

# df = load_csv('data.csv')
# # print(df.loc[1, 'posts'])
# print(f"Known Type: {df.loc[2, 'type']}")
# print(classify(df.loc[2, 'posts']))
# # print(df.info())

create_classification_csv('data.csv', OUTPUT_FILE_NAME+'.csv')

results_summary = OUTPUT_FILE_NAME+'_summary.txt'
with open(results_summary, 'w') as f:
    sys.stdout = f
    calculate_accuracy(OUTPUT_FILE_NAME + '.csv')
    sys.stdout = sys.__stdout__

