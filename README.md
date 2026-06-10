# Akıllı Geri Alma Artırılmış Üretim: Büyük Dil Modellerinde Bağlamsal Hassasiyet ve Verimliliği Geliştirmek
# Intelligent Retrieval-Augmented Generation: Enhancing Contextual Sensitivity and Efficiency in Large Language Models

------------

### Yüksek Lisans Tezi - Fırat Üniversitesi (Fen Bilimleri Enstitüsü) - Yazılım Mühendiliği Ana Bilim Dalı


README.md içine tezin başlığını, bölüm numaralarını ve her not defterinin hangi bölüme karşılık geldiğini yazan 10 satırlık bir açıklama yeterli olur.


Türkçe tez metni için:

Deneylerin tam olarak yeniden üretilebilmesi için IntelliRAG'ın kaynak kodu, Google Colab not defterleri ve değerlendirme betikleri aşağıdaki GitHub deposunda kamuya açık biçimde paylaşılmıştır:

🔗 https://github.com/aminaslami/IntelliRAG

Depo; veri ön işleme, indeksleme, hibrit geri alma, yeniden sıralama ve RAGAS değerlendirme adımlarına karşılık gelen ayrı Colab not defterlerini içermektedir. Kullanılan tüm model ağırlıkları HuggingFace Hub üzerinden erişilebilir durumdadır.



    torch==2.2.1+cu118 
sürümünün kurulabilmesi için PyTorch CUDA deposunun belirtilmesi gerekir.


------------------

IntelliRAG/

├── 01_preprocessing.ipynb

├── 02_indexing_faiss_bm25.ipynb

├── 03_hybrid_retrieval.ipynb

├── 04_reranking.ipynb

├── 05_acc_compression.ipynb

├── 06_evaluation_ragas.ipynb

├── requirements.txt

└── README.md


<img width="838" height="281" alt="image" src="https://github.com/user-attachments/assets/9a271704-c6d3-48bc-9885-ec34c62a07e1" />


