# Recommendation Systems:
- Simple Recommendation Sys.
- Association Rule Learning
- Content Based Filtering
- Collaborative(user, item, model)

1- ARL(Association Rule Learning): 
   
   Veri setindeki örüntüleri bulmak için kullanılan kural tabanlı bir makine öğrenmesi tekniğidir. Müşteriler A ve B birlikte ürün X'i satın alıyorlarsa, 
   o zaman ürün Y'i de satın alıyorlar tespitleri yapılabilir. Market analizinde kullanılabilir.
   
   -Apriori Alg.: Kullanım amacı sepet analizi yöntemi ile ürün birlikteliklerini ortaya çıkarma.
                  Apriori algoritması, veri setindeki sık sık birlikte bulunan öğe kümelerini (itemset'leri) bulur ve bu itemset'lerinden güçlü kural setleri oluşturur. 
                  İlk olarak, tek öğelerin kümeleri oluşturulur ve ardından aday itemset'ler oluşturulur. Bu aday itemset'lerin support değerleri hesaplanır ve belirli 
                  bir min_support değerini aşmayan itemset'ler eleme yapılır. Sonra, elemeden geçen itemset'lerden yeni aday itemset'ler oluşturulur ve bu adaylar için
                  de support değerleri hesaplanır. Bu adımlar tekrarlanır ve güçlü kural setleri oluşturulur. Apriori algoritması, veri setindeki büyük ölçekli veri 
                  kümelerinde bile etkili bir şekilde çalışabilir ve sık sık kullanılan öğe kombinasyonlarını keşfederek anlamlı kural setleri oluşturabilir .

   - Supprt(X,Y): Freq(X,Y) / N (x ve y nin birlikte görülme olasılığı)
   - confidence(X,Y): Freq(X,Y) / Freq(X) (X alınınca y'nin alınma olasılığı)
   - Lift(X,Y) :Support(X,Y) / Support(x) * Support(Y) (x satın alınınca Y'nin satın alınma olasılığı lift kadar artar)

2- Content Based Filtering: 
  Ürün içerikleri(meta bilgileri) benzerlikleri üzerinden tavsiyeler geliştirilir.
   
  Metinlerin matematiksel temsili(Count Vector, TF-IDF),
  Benzerlik hesaplama

    - Count Vector: metin belgelerini sayısal vektörlere dönüştürmek için kullanılan bir işlemdir. Her bir belgedeki kelimelerin 
      frekansı hesaplanır ve her belge bu frekanslarla temsil edilir. Bu, metin verilerini makine öğrenimi modellerine 
      girdi olarak verebilmek için kullanılır. Uzaklık temelli benzerlik hesaplar (öclid, cosine similarity).
      Frekansı yüksek kelimeler yanıltıcı olabilir!
  
    -TF-IDF: Kelimelerin hem kendi metinlerinde hem de bütün odaklanılan verideki geçme frekansları üzerinden normalizasyon yapar.
             Oluşturulan kelime matrisini bütün dökümanı göz önüne alarak terimlerin frekanslarını da göz önüne alarak genel bir 
             standardizasyon işlemi yapar. Bu işlem Count Vectorden çıkabilecek yanlılıkları giderir. count vektördeki frekansı 
             yüksek kelimelerin yanıltıcılığı bu standartlaştırma işlemi sayesinde engellenmiş olur.

             - Count vectorizer tablosu oluşturulur(kelime frekansları)
             - Term Freq(TF) tablosu oluşturulur(ilgili filmdeki frekans / toplam unique terim sayısı)
             - IDF(Inverse Document Freq.) oluşturulur
             - TF x IDF matrisleri çarpılır
             - L2 normalizasyonu yapılır (Satırların kare topl. bul, ilgli satırdaki tüm hücreleri bulunan değere böl)

3- Collaborative Filtering : 
   - Item based: Ürün benzerliği üzerinden öneriler yapılır.İçerikle ilgli değildir. örneğin kullanıcılar bir filmi beğendiyse o film ile benzer beğenilme 
                 örüntüsüne sahip olan diğer filmler önerilmek istenir. Kullanıcının beğendiği ürünlere benzer ürünleri önerir.
                 Amazon, bir kullanıcının satın aldığı veya incelediği ürünlere benzer ürünleri önerir.

   - User based: Kullanıcıların birbirlerine benzerlikleri üzerinden ilerler. X kişisi ile aynı filmleri izleyen kişiler bulunur. Bu kitlenin beğenilen 
                 filmlerini X kişisi de beğenir. Benzer kullanıcıların beğendiği ancak hedef kullanıcının henüz deneyimlemediği ürünleri önerir.
                 Netflix, bir kullanıcının izleme geçmişine dayanarak benzer filmleri izleyen kullanıcıların beğendiği filmleri önerir.
               
   - Model based: Model-based matrix factorization, öneri sistemlerinde kullanıcılar ve ürünler arasındaki ilişkileri modellemek için kullanılan güçlü 
                  bir tekniktir. Bu teknik, büyük boyutlu kullanıcı-ürün matrisini daha düşük boyutlu faktör matrislerine ayırarak öneri yapar. Matrix 
                  factorization, özellikle Netflix gibi büyük veri setlerinde etkili olan ve popülerliği artan bir tekniktir.
                  
                  - Rating  = U(user) x V(product)
                  - Matrix factorization, kullanıcı-ürün etkileşimlerini iki daha küçük matrise ayırır: 
                    - Kullanıcı matrisi (U): Her bir kullanıcıyı belirli bir özellik vektörüyle temsil eder.
                    - Ürün matrisi (V): Her bir ürünü belirli bir özellik vektörüyle temsil eder.

                  - Matristeki bolukları doldurmak için user ve productlar için olduğu varsayılan latent featurların ağırlıkları
                    var olan veri üzerinden bulunur ve bu ağırlıklar ile var olmayan gözlemler için tahmin yapılır. Başlangıçta
                    bu matrisler rastegele doldurulur, her iterasyonda hatalı tahminler düzenlenerek Rating matrisine ulaşılmaya
                    çalışılır. Her iterasyonda doğruya yaklaşılır. Bunu Gradient descent kullanarak kayıp fonksiyonunu minimize
                    eder. Türeve dayalıdır. Gradyanın negatifi yönünde gidilince min. değerleri verecek parametreler bulunur.
                    Bu kayıp fonksiyonu genellikle tahmin edilen değerler ( 𝑅^ ) ile gerçek değerler ( 𝑅 ) arasındaki farkın
                    kareler toplamı olarak tanımlanır.

                  - Bu iki matrisi çarptığınızda, orijinal kullanıcı-ürün matrisine yakın bir tahmin matrisi elde edersiniz.
                    kullanıcı ve ürün matrisleri, hata fonksiyonunu minimize eden bir optimizasyon algoritması kullanılarak öğrenilir. 
                    Genellikle kullanılan hata fonksiyonu, gerçek değerlendirmeler ile tahmin edilen değerlendirmeler arasındaki
                    farkların kareler ortalamasıdır (Mean Squared Error - MSE).


- Elde hiç kullanıcısı olmayan sistemlerde content based filtering kullanılmalı(login yok)
- Elde loginli kullanıcı varsa ARL kullanılmalı, sepet mantığı, puan yok.

    
