"""
NLP Ödevi: Haber Başlıklarından Otomatik Özetleme
Transformer tabanlı model (T5) kullanarak CNN/DailyMail veri seti üzerinde özetleme
"""

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class NewsSummarizer:
    def __init__(self, model_name="t5-small", max_input_length=512, max_target_length=128):
        """
        Haber özetleme sistemi başlatıcısı
        
        Args:
            model_name: Kullanılacak model adı
            max_input_length: Maksimum giriş metni uzunluğu
            max_target_length: Maksimum hedef özet uzunluğu
        """
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
        # Tokenizer ve model yükleme
        print(f"Model yükleniyor: {model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    # Eğitim ve test işlemleri kaldırıldı
    
    def generate_summary(self, text):
        """
        Tek bir metin için özet üretme
        
        Args:
            text: Özetlenecek metin
            
        Returns:
            Üretilen özet
        """
        input_text = f"summarize: {text}"
        
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=self.max_target_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary 