# NLP Ödevi: Haber Başlıklarından Otomatik Özetleme

Bu proje, CNN/DailyMail veri seti kullanılarak haber makalelerinden otomatik özet çıkarma işlemini gerçekleştiren bir NLP (Doğal Dil İşleme) uygulamasıdır.

## 📋 Proje Özeti

Bu proje, T5 (Text-to-Text Transfer Transformer) modelini kullanarak haber makalelerinden otomatik özet çıkarma görevini gerçekleştirir. Model, CNN/DailyMail veri setindeki makaleleri ve bunların özetlerini kullanarak eğitilmiştir.

## 🎯 Hedefler

- Haber makalelerinden otomatik özet çıkarma
- T5 modelinin performansını değerlendirme
- ROUGE metrikleri ile özet kalitesini ölçme
- Türkçe dokümantasyon ile proje açıklaması

## 📊 Sonuçlar

### Model Performansı
Model eğitimi tamamlandıktan sonra elde edilen ROUGE skorları:

- **ROUGE-1**: 0.3224
- **ROUGE-2**: 0.0715  
- **ROUGE-L**: 0.2239

Bu skorlar, modelin özetleme görevinde makul bir performans gösterdiğini işaret etmektedir.

## 🏗️ Proje Yapısı

```
NLP_ödevi/
├── main.py                 # Ana çalıştırma dosyası
├── preprocess_data.py      # Veri ön işleme fonksiyonları
├── train_and_evaluate.py   # Model eğitimi ve değerlendirme
├── requirements.txt        # Gerekli kütüphaneler
├── evaluation_results.json # Değerlendirme sonuçları
└── README.md              # Bu dosya
```

## 🔧 Kullanılan Teknolojiler

- **Python 3.x**
- **PyTorch**: Derin öğrenme framework'ü
- **Transformers**: Hugging Face transformers kütüphanesi
- **T5-small**: Özetleme için kullanılan model
- **ROUGE**: Özet kalitesi değerlendirme metriği
- **Datasets**: Hugging Face datasets kütüphanesi

## 📦 Kurulum

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Projeyi çalıştırın:
```bash
python main.py
```

## 🔄 Çalışma Süreci

### 1. Veri Yükleme
- CNN/DailyMail veri seti (3.0.0 versiyonu) yüklenir
- Train, validation ve test setleri ayrılır

### 2. Model Hazırlığı
- T5-small modeli ve tokenizer yüklenir
- Model parametreleri ayarlanır

### 3. Veri Ön İşleme
- Metinler temizlenir (küçük harf, noktalama işaretleri kaldırma)
- Stop words kaldırılır
- Tokenization işlemi gerçekleştirilir
- Maksimum uzunluk sınırlamaları uygulanır

### 4. Model Eğitimi
- 500 örnek ile eğitim gerçekleştirilir
- 3 epoch boyunca eğitim
- Validation seti ile performans izleme

### 5. Değerlendirme
- Test seti üzerinde özetleme
- ROUGE metrikleri hesaplama
- Sonuçların JSON formatında kaydedilmesi

## 📈 Model Parametreleri

- **Model**: T5-small
- **Maksimum giriş uzunluğu**: 512 token
- **Maksimum hedef uzunluğu**: 128 token
- **Eğitim epoch sayısı**: 3
- **Batch size**: 2
- **Learning rate**: 3e-5

## 📊 Veri Seti Bilgileri

- **Kaynak**: CNN/DailyMail veri seti
- **Eğitim örnekleri**: 500 (ilk 500 örnek)
- **Validation örnekleri**: 50
- **Test örnekleri**: 5
- **Giriş**: Haber makaleleri
- **Hedef**: Makale özetleri

## 🎯 Sonuçlar ve Değerlendirme

Proje başarıyla tamamlanmış ve aşağıdaki sonuçlar elde edilmiştir:

- Model eğitimi tamamlandı
- ROUGE skorları hesaplandı
- Örnek özetler üretildi
- Sonuçlar JSON dosyasına kaydedildi

### ROUGE Skorları Analizi:
- **ROUGE-1 (0.3224)**: Tek kelime örtüşmesi - makul seviyede
- **ROUGE-2 (0.0715)**: İki kelime örtüşmesi - geliştirilebilir
- **ROUGE-L (0.2239)**: En uzun ortak alt dizi - kabul edilebilir

## 🔮 Gelecek Geliştirmeler

- Daha büyük veri seti kullanımı
- Farklı model mimarileri deneme
- Hiperparametre optimizasyonu
- Daha uzun eğitim süresi
- Türkçe veri seti ile deneme

## 📝 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.

## 👨‍💻 Geliştirici

Bu proje NLP dersi kapsamında geliştirilmiştir.

---

**Not**: Bu proje, doğal dil işleme alanında özetleme görevlerinin nasıl gerçekleştirileceğini göstermek amacıyla hazırlanmıştır. 

