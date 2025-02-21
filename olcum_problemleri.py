###Yapılacak Konu Başlıkları
# 1-Rating Products (Ürün Derecelendirme) – Puanları analiz edeceğiz.
# 2-Sorting Products (Ürün Sıralama) – En iyi kursları doğru şekilde sıralayacağız.
# 3-Sorting Reviews (Yorum Sıralama) – En faydalı yorumları öne çıkaracağız.
# 4-A/B Testing (A/B Testi) – İki farklı yöntem arasında istatistiksel testler yapacağız.



import pandas as pd  # Veri analizi için
import numpy as np   # Sayısal işlemler için
from scipy.stats import norm
from scipy.stats import ttest_ind


#Veri Setini Okuma
df = pd.read_csv(r"C:\Users\PC\OneDrive\Masaüstü\veribilimi\verisetleri\course_reviews.csv")
print(df.head())

# Veri Setinin İçeriği:
# Rating → Kullanıcının verdiği puan (örn: 5.0, 4.5)
# Timestamp → Kullanıcının değerlendirmeyi yaptığı tarih
# Enrolled → Kullanıcının kursa kayıt olduğu tarih
# Progress → Kursun yüzde kaçını tamamlamış
# Questions Asked → Kullanıcının sorduğu soru sayısı
# Questions Answered → Kullanıcının yanıtladığı soru sayısı





#################################################
#################################################
#Rating Products (Ürün Derecelendirme)
#################################################
#################################################


# Ortalama puanı hesapla
average_rating = df["Rating"].mean()
print(f"Ortalama Kurs Puanı: {average_rating:.2f}")

#################################################
#Bayesian Average Rating (BAR) – Güvenilir Puan Hesaplama
#################################################
def bayesian_average_rating(n, C, m, R):
    """
    n = Kursun toplam değerlendirme sayısı
    C = Genel ortalama rating (datasetin genel ortalaması)
    m = Minimum değerlendirme eşiği (Burada tüm yorumların %10'u)
    R = Kursun kendi rating ortalaması
    """
    return (n / (n + m) * R) + (m / (n + m) * C)

# Genel ortalama puan
C = df["Rating"].mean()

# Minimum değerlendirme eşiği (örneğin tüm yorumların %10'u)
m = df["Rating"].count() * 0.10

# Her puan grubuna ait yorum sayısını alalım
rating_counts = df["Rating"].value_counts()

# Bayesian Average Rating hesaplayalım
bar_scores = {rating: bayesian_average_rating(n=rating_counts[rating], C=C, m=m, R=rating) for rating in rating_counts.index}

# Sonuçları sıralayarak gösterelim
sorted_bar_scores = dict(sorted(bar_scores.items(), key=lambda item: item[0]))

print("Bayesian Ortalama Puanlar:")
print(sorted_bar_scores)





#################################################
#################################################
# Sorting Products (Ürün Sıralama) – En İyi Kursları Bulma
#################################################
#################################################

#Wilson Lower Bound (WLB) Fonksiyonu
#Bu yöntem, kursların güvenilirlik skorunu hesaplayarak en iyileri belirlememize yardımcı olur.

def wilson_lower_bound(pos, total, confidence=0.95):
    """
        Wilson Lower Bound Score hesaplama
        pos = Olumlu yorum sayısı (örneğin, 4 ve 5 yıldız alan yorumlar)
        total = Toplam yorum sayısı
        confidence = Güven aralığı (varsayılan 0.95)
    """

    if total == 0:
        return 0
    z = norm.ppf(1 - (1 - confidence) / 2)
    phat = pos / total
    return (phat + (z ** 2) / (2 * total) - z * np.sqrt((phat * (1 - phat) + (z ** 2) / (4 * total)) / total)) / (1 + (z ** 2) / total)

#En İyi Kursları Belirleme
#Her kurs için WLB skorunu hesaplayarak sıralayalım.

df["Positive_Reviews"] = df["Rating"].apply(lambda x: 1 if x >= 4 else 0)

df["WLB_Score"] = df.apply(lambda x: wilson_lower_bound(x["Positive_Reviews"], 1), axis=1)

# Kursları sıralayalım
sorted_courses = df.sort_values("WLB_Score", ascending=False)

print(sorted_courses[["Rating", "WLB_Score"]].head(10))







#################################################
#################################################
#Sorting Reviews (Yorum Sıralama) – En Faydalı Yorumları Öne Çıkarma
#################################################
#################################################

#Yorum Skoru Hesaplama

df["Helpful_Review"] = df["Questions Answered"] - df["Questions Asked"]

df["Review_WLB"] = df.apply(lambda x: wilson_lower_bound(x["Helpful_Review"], 1), axis=1)

# En faydalı yorumları sıralayalım
sorted_reviews = df.sort_values("Review_WLB", ascending=False)

print(sorted_reviews[["Rating", "Helpful_Review", "Review_WLB"]].head(10))








#################################################
#################################################
#A/B Testing (A/B Testi) – İki Yöntemi Karşılaştırma
#################################################
#################################################


#Son olarak, A/B testi ile iki farklı yöntem arasındaki farkın anlamlı olup olmadığını test edeceğiz.
#H0 (Null Hypothesis): İki yöntem arasında fark yoktur.
#H1 (Alternative Hypothesis): İki yöntem arasında fark vardır



#T-Testi ile A/B Testi

# Örnek olarak, kursları 2 gruba bölelim
group_A = df[df["Rating"] >= 4.5]["Progress"]
group_B = df[df["Rating"] < 4.5]["Progress"]

# T-test uygulayalım
t_stat, p_value = ttest_ind(group_A, group_B)

print(f"T-İstatistiği: {t_stat}")
print(f"P-Değeri: {p_value}")

# Sonucu yorumlayalım
if p_value < 0.05:
    print("İki grup arasında istatistiksel olarak anlamlı bir fark var! (H0 RED)")
else:
    print("İki grup arasında anlamlı bir fark yoktur. (H0 KABUL)")











