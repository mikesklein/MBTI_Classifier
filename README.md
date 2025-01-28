# MBTI Tweet Classifier Using Local LLMs

## Overview
The advent of large language models (LLMs) has revolutionized many aspects of natural language processing, including their potential to analyze personality traits based on textual data. This research project seeks to extend the work of Murphy (2024), which demonstrated the efficacy of GPT-3.5 and GPT-4 in predicting Myers-Briggs Type Indicator (MBTI) personality types using tweets. By leveraging local LLMs powered by Ollama’s infrastructure, this study aims to replicate and expand upon Murphy’s experiment. Specifically, it will investigate whether state-of-the-art local LLMs can achieve similar or improved accuracy in predicting MBTI classifications.

Unlike the original research, this study introduces a dual-layer prediction framework. The first objective is to predict the overall MBTI type for each individual. The second objective focuses on predicting the classifications for each of the four MBTI parameters (Introversion-Extraversion, Intuition-Sensing, Thinking-Feeling, and Judging-Perceiving) independently. Additionally, the study will evaluate and compare the predictive performance of multiple local LLMs, exploring their viability as alternatives to cloud-based solutions.

This research not only contributes to the understanding of local LLMs’ capabilities but also addresses privacy concerns by emphasizing locally hosted models, which mitigate risks associated with transmitting sensitive data to external servers. By testing the efficacy of localized AI solutions in personality prediction, this study offers valuable insights into the scalability and ethical implications of micro-targeting and personalization in digital spaces.

## Methodology

This project implements a system to predict Myers-Briggs Personality Types (MBTI) from textual data, specifically individuals’ last 50 tweets. The system employs local large language models (LLM), using powered by Ollama, to classify text into one of 16 MBTI categories. Below is an outline of the methodology:

1. Data Input and Preprocessing
	•	Input Data: The system reads a CSV file containing individuals’ last 50 tweets (posts) and their known MBTI type (type).
	•	Sampling: To manage computational load, a random sample of 400 rows is selected for prediction, ensuring consistent evaluation using a fixed random seed.

2. Personality Type Classification
	•	Prompt Engineering: The model is tasked to predict MBTI personality types using a structured prompts.

3. Prediction Generation:

The generate function uses the Ollama API to process the input prompt and generate a prediction.
A regex pattern is applied to extract the predicted MBTI type from the model’s response.
If no match is found, the prediction is labeled as “UNKNOWN.”

4. MBTI Dimension Accuracy

Predictions are compared to the known MBTI type, evaluating both the overall prediction and individual dimensions:
	•	Dimensions Evaluated:
	•	E/I: Extraversion vs. Introversion
	•	S/N: Sensing vs. Intuition
	•	T/F: Thinking vs. Feeling
	•	J/P: Judging vs. Perceiving
	•	Boolean Flags: For each dimension, a flag indicates whether the prediction matches the known type.

5. Output and Analysis
	•	Classification Output:
	•	Results are saved to a new CSV file, including the following fields:
	•	Type: Known MBTI type.
	•	Prediction: Model’s prediction.
	•	Correct: Whether the prediction matches the known type.
	•	Flags for dimension-level matches: EvI, SvN, TvF, JvP.
	•	Model: Name of the LLM used (llama3.1:8b).
	•	Accuracy Calculation:
	•	Overall accuracy and accuracy for each dimension are calculated as:

	•	Results are printed for the overall classification and each dimension.

6. Execution Flow
	1.	Loading Data: Input CSV is read into a pandas DataFrame.
	2.	Classification: Each row of tweets is passed through the classification pipeline.
	3.	Result Compilation: Predictions and evaluation metrics are compiled into a DataFrame and saved as a CSV.
	4.	Accuracy Evaluation: Classification accuracy is calculated and displayed for both overall and dimension-specific results.

7. Tools and Libraries
	•	Programming Language: Python
	•	Libraries:
	•	ollama: Interface with the local LLM.
	•	pandas: Data manipulation and storage.
	•	re: Regex for extracting MBTI type.
	•	LLM: llama3.1:8b (local model via Ollama)

8. Ethical Considerations
	•	Local Model Usage: Emphasizes privacy by processing data locally without transmitting sensitive information to external servers.
	•	Transparent Evaluation: Results include dimension-level evaluations to ensure detailed insights into model performance.

This methodology ensures a systematic, reproducible approach to evaluating local LLMs’ ability to predict personality types from textual data. The project contributes to understanding the capabilities of local AI systems for personalized applications while addressing ethical considerations.

## Data Source and Personality Café Dataset

The data used in this project primarily consists of tweets and MBTI personality labels, curated to enable personality type predictions. The dataset includes individuals’ last 50 tweets, paired with their known Myers-Briggs Type Indicator (MBTI) personality types. This structure allows for testing the ability of local LLMs to classify text-based data into MBTI categories accurately.

### Personality Café Dataset

A significant portion of the MBTI data originates from the Personality Café dataset, which has been widely used in personality classification research. This dataset aggregates self-reported MBTI types and textual data from forum posts. Although the dataset was not directly used in this study, it serves as a foundational reference for understanding personality classification from textual data. Key attributes of the Personality Café dataset include:
	•	Self-reported MBTI types: Users provide their personality types, which are used as ground truth for model evaluation.
	•	Variety of text formats: Includes forum discussions, which resemble tweets in informal tone and length.
	•	Data diversity: Covers a wide range of topics, enabling the exploration of linguistic patterns associated with MBTI traits.

## Results:

### First Run:
- classified_results_llama3_1_8b.csv
- Model - llama3.1:8b
- N = 400
- Approach - one request to LLM for classification
- Prompt: 
"Predict what this person’s personality type using the Myers-Briggs Personality Type scale. Provide only the acronym of the personality type you think the person is. Here are their last 50 Tweets:"

Total Accuracy: 49.00%
- EvI Accuracy: 68.25%
- SvN Accuracy: 88.75%
- TvF Accuracy: 77.00%
- JvP Accuracy: 71.75%

## Second Run:
- results_llama3.1:8b_3.csv
- Model - llama3.1:8b
- N = 400
- Approach - one request to LLM for classification
- Prompt: 
"Predict what this person’s personality type using the Myers-Briggs Personality Type scale. Provide only the acronym of the personality type you think the person is. Here are their last 50 Tweets:"

Total Accuracy: 50.75%
EvI Accuracy: 69.75%
SvN Accuracy: 90.00%
TvF Accuracy: 72.50%
JvP Accuracy: 70.75%

## Third Run: 
- Model - phi4
- N = 400
- Approach - system and user prompts. 
- System prompt "You are an expert personality classifier based on the Myers-Briggs Personality Type scale. Your task is to analyze text input and classify personality traits accurately. Do not provide explanations, context, or additional text—respond with the acronym only."
- User prompt Predict what this person’s personality type using the Myers-Briggs Personality Type scale.  In your response, provide only the acronym of the personality type you think the person is. Again, only provide a four letter response based on Myers-Briggs. Here are their last 50 Tweets:

Total Accuracy (predict all four): 62.00%
- EvI Accuracy: 88.75%
- SvN Accuracy: 91.50%
- TvF Accuracy: 84.00%
- JvP Accuracy: 74.75%

## Fourth Run:
- Model - llama3.1:8b
 - N = 400
- Approach - system and user prompts. 
- System prompt "You are an expert personality classifier based on the Myers-Briggs Personality Type scale. Your task is to analyze text input and classify personality traits accurately. Do not provide explanations, context, or additional text—respond with the acronym only."
- User prompt Predict what this person’s personality type using the Myers-Briggs Personality Type scale.  In your response, provide only the acronym of the personality type you think the person is. Again, only provide a four letter response based on Myers-Briggs. Here are their last 50 Tweets:

Total Accuracy (predict all four): 46.75%
- EvI Accuracy: 72.50%
- SvN Accuracy: 90.00%
- TvF Accuracy: 81.50%
- JvP Accuracy: 71.75%
