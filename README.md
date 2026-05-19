# 🍏 EkoRaf: Akıllı Raf Ömrü & Dinamik Fiyatlandırma Sistemi

EkoRaf; marketler ve hızlı teslimat uygulamalarında son kullanma tarihi (SKT) yaklaşan gıda ürünlerinin israf olmasını engellemek amacıyla geliştirilmiş, dış değişken duyarlı, kademeli ve otomatik bir karar destek sistemidir.

Normal şartlarda marketler son günü bekleyip %50 gibi keskin indirimler yaparken, EkoRaf anlık satış hızını ve dış değişkenleri (hava durumu, maaş günleri, yerel etkinlikler) analiz ederek ürünler çöpe gitmeden günler önce **optimum, kademeli ve otomatik** mikro indirim oranları belirler.

---

## 🚀 Canlı Prototip / Demo

Projenin fiziksel dünyayı dijitalde simüle eden dinamik dashboard arayüzüne aşağıdaki bağlantıdan anında erişebilirsiniz:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

👉 **[EkoRaf Canlı Yönetim Panelini Açmak İçin Tıklayın](https://share.streamlit.io/)**

> ⚠️ **Not:** Yukarıdaki buton ve bağlantı adreslerini kendi Streamlit canlı yayın linkinizle güncellemeyi unutmayın!

---

## ✨ Öne Çıkan Özellikler

* **🌿 Sürdürülebilirlik Odaklı KPI'lar:** Panel, yöneticilere sadece finansal kârı göstermez; kurtarılan ürün miktarı üzerinden **Engellenen Gıda İsrafı (Ton)** ve **Engellenen Karbon Ayak İzi ($CO_2$e)** raporlarını anlık sunar.
* **🧠 Yapay Zeka & Proaktif Öneri Motoru:** Ürünlerin son gününü beklemeden risk durumlarını analiz eder. Örneğin; *"Bugün hava yağmurlu, mangallık etlerin satış hızı %40 düştü. Fiyatı hemen şimdi %12 düşür ve mobil uygulamada 'Fırsat Ürünü' olarak öne çıkar"* aksiyonunu otomatik üretir.
* **🔮 Dijital İkiz & Senaryo Simülatörü:** Mağaza yöneticileri **"Hafta sonu kar yağarsa stok erime hızım ne olur?"** senaryosunu fiziksel dünyayı dijitalde simüle ederek görebilir ve lojistik önlemler alabilir.

---

## 📊 Veri Kümesi Yapısı

Bu prototip, gerçekçi tüketici davranışları ve simüle edilmiş dış etken öznitelikleri içeren özelleştirilmiş bir supermarket veri kümesi (`SuperMarket Analysis_with_synthetic_features.csv`) tabanında çalışmaktadır. Projede kullanılan temel değişkenler:
* `SKT_Kalan_Gun`: Ürünün tazeliğini ve kalan raf ömrünü temsil eder.
* `Hava_Durumu`: Satış hızını doğrudan etkileyen dış değişken (Karlı, Yağmurlu, Güneşli vb.).
* `Uygulanan_Indirim_Orani`: EkoRaf algoritmasının belirlediği kademeli indirim yüzdesi.

---

## 🛠️ Yerel Kurulum (Local Setup)

Eğer projeyi kendi bilgisayarınızda çalıştırmak isterseniz:

1. Bu repository'i klonlayın:
   ```bash
   git clone [https://github.com/KULLANICI_ADINIZ/ekoraf-dashboard.git](https://github.com/KULLANICI_ADINIZ/ekoraf-dashboard.git)
   cd ekoraf-dashboard
2. Gerekli kütüphaneleri yükleyin:
pip install -r requirements.txt
3. Uygulamayı başlatın:
streamlit run app.py
---
---

## 🏗️ Sistem Mimarisi (System Architecture)

EkoRaf'ın teknik çalışma prensibi ve veri akış mimarisi aşağıdaki şekilde tasarlanmıştır:

```text
[Veri Kaynağı: Excel / CSV] 
       │
       ▼
[EkoRaf Dinamik Karar Motoru (Python)] ───► (Dış Değişken Analizi: Hava Durumu, SKT)
       │
       ▼
[Kademeli Optimize Fiyatlandırma] ─────► (Algoritma Tabanlı Mikro İndirim Hesaplama)
       │
       ▼
[Kullanıcı Paneli (Streamlit Cloud)] ───► (Yönetici ve Mağaza Müdürü Dijital İkizi)
       │
       ▼
[Mobil API Entegrasyonu (Gelecek)] ─────► (Müşteriye Anlık Mobil Bildirim / Push)

## 👥 Ekip (Contributors)

* **Hiranur** - https://github.com/Hiranursozeri
* **Mahmut** - https://github.com/MahmutDnl
---

💡 *Bu proje, sürdürülebilir bir gelecek ve sıfır atık vizyonunu desteklemek amacıyla geliştirilmiştir.*
