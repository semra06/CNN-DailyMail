🧠 NLP Assignment: Automatic Summarization from News Headlines

This project is a Natural Language Processing (NLP) application that performs automatic summarization of news articles using the CNN/DailyMail dataset.
📋 Project Summary

This project performs automatic summarization of news articles using the T5 (Text-to-Text Transfer Transformer) model. The model is trained on the articles and summaries from the CNN/DailyMail dataset.
🎯 Objectives

    Automatically summarize news articles

    Evaluate the performance of the T5 model

    Measure summary quality using ROUGE metrics

    Provide documentation in Turkish

📊 Results
Model Performance

After completing the training, the obtained ROUGE scores are:

    ROUGE-1: 0.3224

    ROUGE-2: 0.0715

    ROUGE-L: 0.2239

These scores indicate a reasonable performance of the model in the summarization task.
🏗️ Project Structure

NLP_ödevi/
├── main.py                 # Main script to run the project
├── preprocess_data.py      # Data preprocessing functions
├── train_and_evaluate.py   # Model training and evaluation
├── requirements.txt        # Required libraries
├── evaluation_results.json # Evaluation results
└── README.md              # This file

🔧 Technologies Used

    Python 3.x

    PyTorch: Deep learning framework

    Transformers: Hugging Face Transformers library

    T5-small: Pretrained model used for summarization

    ROUGE: Metric for evaluating summary quality

    Datasets: Hugging Face Datasets library

📦 Installation

    Install the required libraries:

pip install -r requirements.txt

    Run the project:

python main.py

🔄 Workflow
1. Data Loading

    Load CNN/DailyMail dataset (version 3.0.0)

    Split into train, validation, and test sets

2. Model Setup

    Load the T5-small model and tokenizer

    Set model parameters

3. Data Preprocessing

    Clean text (lowercasing, remove punctuation)

    Remove stop words

    Apply tokenization

    Apply maximum length limits

4. Model Training

    Train using 500 examples

    Train for 3 epochs

    Monitor performance with validation set

5. Evaluation

    Summarize on the test set

    Compute ROUGE metrics

    Save results in JSON format

📈 Model Parameters

    Model: T5-small

    Max input length: 512 tokens

    Max target length: 128 tokens

    Number of epochs: 3

    Batch size: 2

    Learning rate: 3e-5

📊 Dataset Information

    Source: CNN/DailyMail dataset

    Training examples: 500 (first 500 entries)

    Validation examples: 50

    Test examples: 5

    Input: News articles

    Target: Article summaries

🎯 Results & Evaluation

The project was successfully completed with the following outcomes:

    Model training completed

    ROUGE scores calculated

    Example summaries generated

    Results saved in a JSON file

ROUGE Score Analysis:

    ROUGE-1 (0.3224): Unigram overlap — moderate performance

    ROUGE-2 (0.0715): Bigram overlap — room for improvement

    ROUGE-L (0.2239): Longest common subsequence — acceptable

🔮 Future Improvements

    Use a larger dataset

    Experiment with different model architectures

    Hyperparameter tuning

    Extend training duration



