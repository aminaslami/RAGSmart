# IntelliRAG

## Akıllı Geri Alma Artırılmış Üretim: Büyük Dil Modellerinde Bağlamsal Hassasiyet ve Verimliliği Geliştirmek

**Intelligent Retrieval-Augmented Generation: Enhancing Contextual Sensitivity and Efficiency in Large Language Models**

---

> **Yüksek Lisans Tezi**  
> Fırat Üniversitesi, Fen Bilimleri Enstitüsü  
> Yazılım Mühendiliği Ana Bilim Dalı  
> **Yazar:** Mohammad Amin ASLAMI
> **Danışman:** Doç. Dr. Ferhat UÇAR
> **Yıl:** 2026

---

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Mimari](#mimari)
- [Depo Yapısı ve Bölüm Eşleşmesi](#depo-yapısı-ve-bölüm-eşleşmesi)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Yeniden Üretilebilirlik](#yeniden-üretilebilirlik)
- [Deneysel Sonuçlar](#deneysel-sonuçlar)
- [Atıf](#atıf)
- [Lisans](#lisans)

---

## Genel Bakış

**IntelliRAG**, büyük dil modeli (LLM) tabanlı uygulamalarda bilgi geri alma sürecini uçtan uca optimize etmek amacıyla tasarlanmış bütünleşik bir RAG (Retrieval-Augmented Generation) çerçevesidir. Çerçeve dört tamamlayıcı modülü bir araya getirmektedir:

| Modül | Açıklama | İlgili Tez Bölümü |
|---|---|---|
| **Dinamik Semantik Parçalama** | Sabit uzunluk yerine anlamsal sınırlara dayalı belge bölümleme | Bölüm 4.1 |
| **Hibrit Yoğun-Seyrek Geri Alma** | FAISS (yoğun) + BM25 (seyrek) birleşimi | Bölüm 4.2 |
| **Çapraz Kodlayıcı Yeniden Sıralama** | Geri alınan pasajların alaka düzeyine göre yeniden sıralanması | Bölüm 4.3 |
| **Uyarlanabilir Bağlam Sıkıştırma (ACC)** | Token düzeyinde önem puanlamasıyla bağlam uzunluğu optimizasyonu | Bölüm 4.4 |

Natural Questions (NQ) kıyaslama veri kümesi üzerinde gerçekleştirilen deneysel değerlendirmede IntelliRAG, tüm metriklerde istatistiksel olarak anlamlı iyileştirmeler elde etmiştir: sadakat %71 → %91, bağlam hassasiyeti %68 → %89, yanıt alaka düzeyi %73 → %92.

---

## Mimari


<img width="581" height="432" alt="image" src="https://github.com/user-attachments/assets/108e7a61-475e-44c1-95d2-f6e4cfe5b684" />

Şekil 1 - RAG iş akışı ve mimarisi. Kullanıcı sorgusu (Bölüm 3.3)

---

## Depo Yapısı ve Bölüm Eşleşmesi

Aşağıdaki tablo, depodaki her dosyanın tezdeki hangi bölüm ve deneyle ilişkili olduğunu göstermektedir.

```
IntelliRAG/
├── notebooks
    └── 01_IntelliRAG_Pipeline.ipynb        # Ana pipeline — Bölüm 4 (tam sistem)
    └── 02_IntelliRAG_Ablation.ipynb        # Ablasyon çalışmaları — Bölüm 5 (bulgular ve tartışma)
├── scripts
    └── evaluate.py
├── README.md
├── Seed_Tohum(42).py                       # Yeniden üretilebilirlik — Bölüm 4.6.1
└── requirements.txt                        # Kullanılan kütüphane sürümleri — Bölüm 4.6

├── notebooks/
│   ├── 01_IntelliRAG_Pipeline.ipynb   # Tam pipeline demosu
│   └── 02_IntelliRAG_Ablation.ipynb   # Ablation çalışması
├── scripts/
│   └── evaluate.py                     # Komut satırı değerlendirme
├── data/                               # Örnek veriler (otomatik indirilir)
├── results/                            # Çıktı grafikleri ve JSON
├── requirements.txt                    # pip bağımlılıkları
├── environment.yml                     # conda ortamı
└── README.md

```
---

## Kurulum

### Gereksinimler

- Python ≥ 3.9
- CUDA destekli GPU (önerilir) veya Google Colab Pro

### Adımlar

```bash
# 1. Depoyu klonlayın
git clone https://github.com/aminaslami/IntelliRAG.git
cd IntelliRAG

# 2. Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### Temel Bağımlılıklar - Kütüphane Versiyonları
Versiyonlar ya eşit yada büyük olabilir

```

Python>=3.10.12
transformers>=4.40.0
sentence-transformers>=2.7.0
faiss-cpu>=1.8.0
rank-bm25>=0.2.2
bitsandbytes>=0.43.1
numpy>=1.26.4
datasets>=2.19.0
ragas>=0.1.7

```

> **Not:** GPU ortamı için `faiss-cpu` yerine `faiss-gpu` kullanılması önerilir.

Sürümünün kurulabilmesi için PyTorch CUDA deposunun kurulması gerekiyor.

    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
---

## Kullanım

### Google Colab

`01_IntelliRAG_Pipelinen.ipynb ve 02_IntelliRAG_Ablation` dosyasını doğrudan Google Colab üzerinde açarak çalıştırabilirsiniz. Not defterilerinde, her bölümde açıklayıcı başlıklar ve hücre açıklamaları içermektedir.

### Yerel Ortam

```bash
jupyter notebook 01_IntelliRAG_Pipelinen.ipynb

jupyter notebook 02_IntelliRAG_Ablation

```

---

## Yeniden Üretilebilirlik

Tezde raporlanan tüm sonuçlar (`Tablo 5.1–5.5`) aşağıdaki sabit tohum değerleriyle üretilmiştir. Her konfigürasyon için 3 bağımsız koşunun ortalaması alınmış; güven aralıkları bootstrap yöntemiyle hesaplanmıştır.

```python
import torch, numpy as np, random
from transformers import set_seed

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)
```

Tam tohum yapılandırması için bkz. [`Seed_Tohum(42).py`](./Seed_Tohum(42).py).

---

## Deneysel Sonuçlar

Aşağıdaki tablo, IntelliRAG'ın NQ doğrulama kümesi (N = 7.830) üzerindeki temel konfigürasyonlarla karşılaştırmalı başarımını özetlemektedir.

| Sistem | Sadakat | Bağlam Hassasiyeti | Yanıt Alaka Düzeyi | Gecikme (ms) |
|---|---|---|---|---|
| Temel RAG (salt yoğun) | 0,71 | 0,68 | 0,73 | 480 |
| + Dinamik Parçalama | 0,81 | 0,79 | 0,82 | 495 |
| + Hibrit Geri Alma | 0,85 | 0,83 | 0,86 | 520 |
| + Yeniden Sıralama | 0,88 | 0,86 | 0,89 | 560 |
| **IntelliRAG (Tam)** | **0,91** | **0,89** | **0,92** | **610** |

Tüm karşılaştırmalar istatistiksel olarak anlamlıdır (p < 0,001; Cohen's d > 0,95).  
Ablasyon çalışmaları için bkz. Tablo 5.2–5.5 (tez, Bölüm 5).

---

## Atıf

Bu çalışmayı araştırmanızda kullanıyorsanız lütfen aşağıdaki biçimde atıf yapınız:

```bibtex
@mastersthesis{aslami2026intellirag,
  author    = {Aslami. Mohammad Amin, Uçar. Ferhat},
  title     = {Intelligent Retrieval-Augmented Generation: Enhancing
               Contextual Sensitivity and Efficiency in Large Language Models},
  school    = {Fırat Üniversitesi, Fen Bilimleri Enstitüsü},
  year      = {2026},
  type      = {Yüksek Lisans Tezi},
  url       = {https://github.com/aminaslami/IntelliRAG}
}
```

---

## Lisans

Bu depo akademik, bilimsel ve eğitim amaçlarla kamuya açık olarak paylaşılmaktadır. 

Ticari kullanıma açıktır.

---

*Bu depo, Fırat Üniversitesi Fen Bilimleri Enstitüsü'ne sunulan yüksek lisans tezinin deneysel kodlarını ve yeniden üretilebilirlik materyallerini içermektedir.*
