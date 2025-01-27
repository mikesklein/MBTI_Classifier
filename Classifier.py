from ollama import chat
from ollama import ChatResponse
import re
import pandas as pd
import sys
import time

MODEL = 'llama3.1:8b'
INPUT_FILE = 'data.csv'
RUN = 5
SAMPLE_SIZE = 400
OUTPUT_FILE_NAME = f"results_{MODEL}_{RUN}"

"""
A script to classify tweets based on the Myers-Briggs Personality Type scale. You need Ollama installed locally to run this script.
"""
# def generate(prompt, data):
#     #
#     """
#     Give the model a task and content, generate a response from the model and return the result.
#     """
#     response: ChatResponse = chat(model=MODEL, stream=False, messages=[
#       {
#         'role': 'user',
#         'content': f"{prompt}{data}"
#       },
#     ])
#     return strip_think_tags(response.message.content)


def generate(prompt, data):
    """
    Give the model a task and content, generate a response from the model, and return the result.
    Handles cases where data is too long by splitting it into multiple messages.
    """
    MAX_TOKENS = 2048  # The token limit for the model
    PROMPT_TOKEN_ESTIMATE = 50  # Estimated tokens for the prompt
    CHUNK_SIZE = MAX_TOKENS - PROMPT_TOKEN_ESTIMATE - 50  # Leave room for response and tags

    # Function to split data into chunks
    def split_into_chunks(data, chunk_size):
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    # Split data into manageable chunks
    data_chunks = split_into_chunks(data, CHUNK_SIZE)

    # Construct the messages thread
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]  # Optional system message
    messages.append({'role': 'user', 'content': prompt})

    for chunk in data_chunks:
        messages.append({'role': 'user', 'content': chunk})

    # Generate response
    response: ChatResponse = chat(model=MODEL, stream=False, messages=messages)
    return strip_think_tags(response.message.content)

def strip_think_tags(response_text):
    """
    Strips everything between <think> and </think> tags from the response if using DeepSeek
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

def classify_by_preference(data):
    """
    Classify the data using by preference.
    """
    # use the generate function to classify the data into the MTBI categories
    mbti_regex = r'^[IENSTFJP]$'
    #classify I vs E
    # prompt = f"Predict whether this person has a preference for Introversion or Extroversion using Myers-Briggs Personality Type scale.  Respond only with an I for Introversion or E for Extroversion. Again only provide I or E. Here are their last 50 Tweets:"
    prompt = f"Predict if this person is an Extravert (E) or Introvert (I) using the Myers-Briggs Personality Type scale. An Extravert (E) is energized by social interactions, outwardly focused, and enjoys engaging with the external world. An Introvert (I) is energized by reflection, inwardly focused, and prefers deep, meaningful interactions or solitary activities. Provide only E for Extroversion or I for Introversion. Only respond with one letter. Here are their last 50 Tweets:"
    response = generate(prompt, data).strip()
    match = re.search(mbti_regex, response)
    IvE = match.group(0) if match else "-"

    # prompt = f"Predict whether this person has a preference for Intuition or Sensing using Myers-Briggs Personality Type scale.  Respond only with an N for Intuition or S for Sensing. Again only provide N or S. Here are their last 50 Tweets:"
    prompt = f"Predict if this person is an Intuitive (N) or Sensing (S) type using the Myers-Briggs Personality Type scale. An Intuitive (N) type prefers abstract, big-picture thinking, focusing on ideas and future possibilities. A Sensing (S) type relies on concrete, detail-oriented thinking and prefers practical, present-focused information. Provide only N for Intuitive or S for Sensing. Only respond with one letter. Here are their last 50 Tweets:"
    response = generate(prompt, data).strip()
    match = re.search(mbti_regex, response)
    NvS = match.group(0) if match else "-"

    # prompt = f"Predict whether this person has a preference for Thinking or Feeling using Myers-Briggs Personality Type scale.  Respond only with a T for Thinking or F for Feeling. Again only provide T or F. Here are their last 50 Tweets:"
    prompt = f"Predict if this person is a Thinking (T) or Feeling (F) type using the Myers-Briggs Personality Type scale. A Thinking (T) type makes decisions based on logic, objective analysis, and fairness. A Feeling (F) type makes decisions based on empathy, personal values, and harmony. Provide only T for Thinking or F for Feeling. Only respond with one letter. Here are their last 50 Tweets:"
    response = generate(prompt, data).strip()
    match = re.search(mbti_regex, response)
    TvF = match.group(0) if match else "-"

    # prompt = f"Predict whether this person has a preference for Judging or Perceiving using Myers-Briggs Personality Type scale.  Respond only with a J for Judging or P for Perceiving. Again only provide J or P. Here are their last 50 Tweets:"
    prompt = f"Predict if this person is a Judging (J) or Perceiving (P) type using the Myers-Briggs Personality Type scale. A Judging (J) type prefers structure, planning, and organization, and likes to have decisions made. A Perceiving (P) type prefers flexibility, adaptability, and spontaneity, and likes to keep their options open. Provide only J for Judging or P for Perceiving. Only respond with one letter.Here are their last 50 Tweets:"
    response = generate(prompt, data).strip()
    match = re.search(mbti_regex, response)
    JvP = match.group(0) if match else "-"

    return IvE + NvS + TvF + JvP


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

        # predicted_type = classify(row['posts'])  # classify function assumed to exist
        predicted_type = classify_by_preference(row['posts'])
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
        # time.sleep(30)

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

