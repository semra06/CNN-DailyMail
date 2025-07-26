import torch
import numpy as np
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from rouge_score import rouge_scorer
from tqdm import tqdm
import json

def compute_metrics(eval_preds, tokenizer):
    """
    ROUGE metriklerini hesaplayan fonksiyon
    """
    predictions, labels = eval_preds
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Initialize ROUGE scorer
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate ROUGE scores
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for pred, label in zip(decoded_preds, decoded_labels):
        scores = rouge_scorer_obj.score(label, pred)
        rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
    
    # Calculate averages
    result = {
        'rouge1': np.mean(rouge_scores['rouge1']),
        'rouge2': np.mean(rouge_scores['rouge2']),
        'rougeL': np.mean(rouge_scores['rougeL'])
    }
    
    return result

def train_and_evaluate(model, tokenizer, train_dataset, eval_dataset, test_dataset, model_name="t5-small", epochs=3):
    """
    Modeli eğitir, değerlendirir ve sonuçları kaydeder.
    """
    # 1. Model eğitimi
    print("\n5. Model eğitimi başlıyor...")
    training_args = Seq2SeqTrainingArguments(
        output_dir="./model_output",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=epochs,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        warmup_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        # TensorBoard ve diğer logging'i tamamen devre dışı bırak
        report_to=[],
        logging_dir=None,
        # Logging'i dosyaya yazma
        logging_first_step=False,
        logging_strategy="no",
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer)
    )
    trainer.train()
    trainer.save_model("./model_output/best_model")
    print("Model eğitimi tamamlandı!")

    # 2. Model değerlendirmesi
    print("\n6. Model değerlendirmesi başlıyor...")
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = []
    examples = []
    for i, example in enumerate(tqdm(test_dataset, desc="Değerlendirme")):
        input_text = f"summarize: {example['article']}"
        inputs = tokenizer(
            input_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=128,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        predicted_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        actual_summary = example['highlights']
        scores = rouge_scorer_obj.score(actual_summary, predicted_summary)
        rouge_scores.append(scores)
        examples.append({
            'input': example['article'][:200] + "...",
            'actual_summary': actual_summary,
            'predicted_summary': predicted_summary,
            'rouge_scores': {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        })
    print("\n" + "="*80)
    print("MODEL DEĞERLENDİRME SONUÇLARI")
    print("="*80)
    avg_rouge1 = np.mean([ex['rouge_scores']['rouge1'] for ex in examples])
    avg_rouge2 = np.mean([ex['rouge_scores']['rouge2'] for ex in examples])
    avg_rougeL = np.mean([ex['rouge_scores']['rougeL'] for ex in examples])
    print(f"\nOrtalama ROUGE Skorları:")
    print(f"ROUGE-1: {avg_rouge1:.4f}")
    print(f"ROUGE-2: {avg_rouge2:.4f}")
    print(f"ROUGE-L: {avg_rougeL:.4f}")
    print(f"\nÖrnek Çıktılar ({len(examples)} adet):")
    print("-"*80)
    for i, example in enumerate(examples, 1):
        print(f"\nÖRNEK {i}:")
        print(f"Giriş Metni: {example['input']}")
        print(f"Gerçek Özet: {example['actual_summary']}")
        print(f"Tahmin Edilen Özet: {example['predicted_summary']}")
        print(f"ROUGE-1: {example['rouge_scores']['rouge1']:.4f}")
        print(f"ROUGE-2: {example['rouge_scores']['rouge2']:.4f}")
        print(f"ROUGE-L: {example['rouge_scores']['rougeL']:.4f}")
        print("-"*40)
    results = {
        'model_info': {
            'model_name': model_name,
            'max_input_length': 512,
            'max_target_length': 128,
            'epochs': epochs,
            'train_samples': len(train_dataset),
            'eval_samples': len(eval_dataset)
        },
        'examples': examples,
        'average_scores': {
            'rouge1': avg_rouge1,
            'rouge2': avg_rouge2,
            'rougeL': avg_rougeL
        }
    }
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSonuçlar 'evaluation_results.json' dosyasına kaydedildi.")
    print("\nÖdev tamamlandı! ✅") 