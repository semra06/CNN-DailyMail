"""
NLP Ödevi: Haber Başlıklarından Otomatik Özetleme
Ana çalıştırma dosyası
"""

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from rouge_score import rouge_scorer
from tqdm import tqdm
import json
import warnings
from preprocess_data import preprocess_dataset
from train_and_evaluate import train_and_evaluate
warnings.filterwarnings('ignore')

def main():
    """Ana fonksiyon - NLP özetleme ödevini çalıştırır"""
    print("="*60)
    print("NLP Ödevi: Haber Başlıklarından Otomatik Özetleme")
    print("="*60)
    
    # 1. Veri setini yükle
    print("\n1. CNN/DailyMail veri seti yükleniyor...")
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    print(f"Veri seti yüklendi! Train: {len(dataset['train'])} örnek")
    
    # 2. Model ve tokenizer yükle
    print("\n2. T5-small modeli yükleniyor...")
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # 3. Veri ön işleme
    print("\n3. Veri ön işleme başlıyor...")
    processed_dataset = preprocess_dataset(
        dataset,
        input_column="article",
        target_column="highlights",
        tokenizer=tokenizer,
        max_input_length=512,
        max_target_length=128
    )
    print("Veri ön işleme tamamlandı!")
    
    # 4. Veri setini böl (ilk 500 örnek kullan)
    print("\n4. Veri seti bölünüyor...")
    train_size = min(500, len(processed_dataset["train"]))
    eval_size = min(50, len(processed_dataset["validation"]))
    
    train_dataset = processed_dataset["train"].select(range(train_size))
    eval_dataset = processed_dataset["validation"].select(range(eval_size))
    
    # Test için orijinal veri setini kullan (preprocessing yapmadan)
    test_dataset = dataset["test"].select(range(5))  # Test için 5 örnek
    
    print(f"Eğitim veri seti: {len(train_dataset)} örnek (ilk 500)")
    print(f"Değerlendirme veri seti: {len(eval_dataset)} örnek")
    print(f"Test veri seti: {len(test_dataset)} örnek (orijinal veri)")
    
    # 5. Model eğitimi ve değerlendirme
    train_and_evaluate(
        model, tokenizer, train_dataset, eval_dataset, test_dataset,
        model_name=model_name, epochs=3
    )
    
    print("\nÖdev tamamlandı! ✅")

if __name__ == "__main__":
    main() 