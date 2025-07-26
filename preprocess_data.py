import re
import string
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text_en(text):
    """
    Clean English text: lowercase, remove punctuation, extra spaces, non-ASCII chars, and remove stopwords.
    """
    if text is None:
        return ""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove non-ascii characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def preprocess_dataset(dataset, input_column="article", target_column="highlights", tokenizer=None, max_input_length=512, max_target_length=128):
    """
    Preprocess dataset for summarization: clean and tokenize.
    """
    def tokenize_function(examples):
        inputs = [clean_text_en(text) for text in examples[input_column]]
        targets = [clean_text_en(text) for text in examples[target_column]]
        model_inputs = tokenizer(
            [f"summarize: {inp}" for inp in inputs],
            max_length=max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["labels"][model_inputs["labels"] == tokenizer.pad_token_id] = -100
        return model_inputs
    processed_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    return processed_dataset 