MBTI Tweet Classifier Using Local LLMs

The advent of large language models (LLMs) has revolutionized many aspects of natural language processing, including their potential to analyze personality traits based on textual data. This research project seeks to extend the work of Murphy (2024), which demonstrated the efficacy of GPT-3.5 and GPT-4 in predicting Myers-Briggs Type Indicator (MBTI) personality types using tweets. By leveraging local LLMs powered by Ollama’s infrastructure, this study aims to replicate and expand upon Murphy’s experiment. Specifically, it will investigate whether state-of-the-art local LLMs, such as llama3.1:8b, can achieve similar or improved accuracy in predicting MBTI classifications.

Unlike the original research, this study introduces a dual-layer prediction framework. The first objective is to predict the overall MBTI type for each individual. The second objective focuses on predicting the classifications for each of the four MBTI parameters (Introversion-Extraversion, Intuition-Sensing, Thinking-Feeling, and Judging-Perceiving) independently. Additionally, the study will evaluate and compare the predictive performance of multiple local LLMs, exploring their viability as alternatives to cloud-based solutions.

This research not only contributes to the understanding of local LLMs’ capabilities but also addresses privacy concerns by emphasizing locally hosted models, which mitigate risks associated with transmitting sensitive data to external servers. By testing the efficacy of localized AI solutions in personality prediction, this study offers valuable insights into the scalability and ethical implications of micro-targeting and personalization in digital spaces.

Methodology

This project implements a system to predict Myers-Briggs Personality Types (MBTI) from textual data, specifically individuals’ last 50 tweets. The system employs a local large language model (LLM), llama3.1:8b, powered by Ollama, to classify text into one of 16 MBTI categories. Below is an outline of the methodology:

1. Data Input and Preprocessing
	•	Input Data: The system reads a CSV file containing individuals’ last 50 tweets (posts) and their known MBTI type (type).
	•	Sampling: To manage computational load, a random sample of 400 rows is selected for prediction, ensuring consistent evaluation using a fixed random seed.

2. Personality Type Classification
	•	Prompt Engineering: The model is tasked to predict MBTI personality types using a structured prompt:

Predict what this person’s personality type using the Myers-Briggs Personality Type scale. 
Provide only the acronym of the personality type you think the person is. 
Here are their last 50 Tweets:


	•	Prediction Generation:
	•	The generate function uses the Ollama API to process the input prompt and generate a prediction.
	•	A regex pattern is applied to extract the predicted MBTI type from the model’s response.
	•	If no match is found, the prediction is labeled as “UNKNOWN.”

3. MBTI Dimension Accuracy

Predictions are compared to the known MBTI type, evaluating both the overall prediction and individual dimensions:
	•	Dimensions Evaluated:
	•	E/I: Extraversion vs. Introversion
	•	S/N: Sensing vs. Intuition
	•	T/F: Thinking vs. Feeling
	•	J/P: Judging vs. Perceiving
	•	Boolean Flags: For each dimension, a flag indicates whether the prediction matches the known type.

4. Output and Analysis
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

5. Execution Flow
	1.	Loading Data: Input CSV is read into a pandas DataFrame.
	2.	Classification: Each row of tweets is passed through the classification pipeline.
	3.	Result Compilation: Predictions and evaluation metrics are compiled into a DataFrame and saved as a CSV.
	4.	Accuracy Evaluation: Classification accuracy is calculated and displayed for both overall and dimension-specific results.

6. Tools and Libraries
	•	Programming Language: Python
	•	Libraries:
	•	ollama: Interface with the local LLM.
	•	pandas: Data manipulation and storage.
	•	re: Regex for extracting MBTI type.
	•	LLM: llama3.1:8b (local model via Ollama)

7. Ethical Considerations
	•	Local Model Usage: Emphasizes privacy by processing data locally without transmitting sensitive information to external servers.
	•	Transparent Evaluation: Results include dimension-level evaluations to ensure detailed insights into model performance.

This methodology ensures a systematic, reproducible approach to evaluating local LLMs’ ability to predict personality types from textual data. The project contributes to understanding the capabilities of local AI systems for personalized applications while addressing ethical considerations.



Results:

