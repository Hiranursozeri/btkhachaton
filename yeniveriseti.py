import pandas as pd
import numpy as np

# CSV dosyasını oku
df = pd.read_csv('SuperMarket Analysis.csv')

print(f"Orijinal veri seti şekli: {df.shape}")
print(f"\nMevcut kolonlar: {df.columns.tolist()}")
print(f"\nİlk 5 satır:\n{df.head()}")

# ============================================
# SENTETİK KOLONLAR EKLE
# ============================================

# 1. SKT_Kalan_Gun (Mantıklı Dağılım)
skt_values = []
for product_line in df['Product line']:
    if product_line == 'Food and beverages':
        # %80 ihtimalle kısa (1-7 gün), %20 ihtimalle uzun (8-30 gün)
        if np.random.random() < 0.8:
            skt_values.append(np.random.randint(1, 8))
        else:
            skt_values.append(np.random.randint(8, 31))
    else:
        # Diğer ürünler: 1-30 gün eşit dağılım
        skt_values.append(np.random.randint(1, 31))

df['SKT_Kalan_Gun'] = skt_values

# 2. Hava_Durumu (Dış Faktör)
df['Hava_Durumu'] = np.random.choice(['Güneşli', 'Yağmurlu', 'Bulutlu', 'Karlı'], len(df))

# 3. Uygulanan_Indirim_Orani (SKT ile negatif korelasyon)
# Formül: SKT azaldıkça indirim artar
indirim_values = []
for skt_gun in df['SKT_Kalan_Gun']:
    # Temel hesap: (30 - SKT) × 1.67 + rastgele gürültü
    base_discount = (30 - skt_gun) * 1.67
    noise = np.random.uniform(-5, 5)  # ±5% rastgele varyasyon
    discount = max(0, min(50, base_discount + noise))  # 0-50% aralığında sınırla
    indirim_values.append(round(discount, 2))

df['Uygulanan_Indirim_Orani'] = indirim_values

# ============================================
# SONUÇLARI GÖSTER
# ============================================
print("\n" + "="*60)
print("📊 YENİ VERİ SETİ ÖZET:")
print("="*60)
print(f"Satır sayısı: {len(df)}")
print(f"Toplam kolon sayısı: {len(df.columns)}")

print(f"\n✨ Yeni eklenen 3 kolon:")
print(df[['SKT_Kalan_Gun', 'Hava_Durumu', 'Uygulanan_Indirim_Orani']].head(10))

print(f"\n📈 İstatistikler:")
print(df[['SKT_Kalan_Gun', 'Uygulanan_Indirim_Orani']].describe())

# Korelasyon kontrol et
correlation = df['SKT_Kalan_Gun'].corr(df['Uygulanan_Indirim_Orani'])
print(f"\n🔗 SKT_Kalan_Gun ile Indirim Korelasyonu: {correlation:.3f}")
print(f"   (Negatif = SKT azaldıkça indirim artar ✓)")

# Food and Beverages kontrolü
food_avg_skt = df[df['Product line'] == 'Food and beverages']['SKT_Kalan_Gun'].mean()
other_avg_skt = df[df['Product line'] != 'Food and beverages']['SKT_Kalan_Gun'].mean()
print(f"\n🍔 Ortalama SKT - Food & Beverages: {food_avg_skt:.1f} gün")
print(f"   Ortalama SKT - Diğer Ürünler: {other_avg_skt:.1f} gün")

# CSV'ye kaydet
output_file = 'SuperMarket Analysis_with_synthetic_features.csv'
df.to_csv(output_file, index=False, encoding='utf-8')
print(f"\n✅ Yeni dosya kaydedildi: {output_file}")
print("="*60)