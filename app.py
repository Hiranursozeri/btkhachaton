import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ==========================================
# SAYFA YAPILANDIRMASI & TASARIM
# ==========================================
st.set_page_config(
    page_title="EkoRaf | Akıllı Raf Ömrü & Dinamik Fiyatlandırma",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title { font-size:36px !important; font-weight: bold; color: #2E7D32; text-align: center; margin-bottom: 5px; }
    .subtitle { font-size:16px !important; color: #555555; text-align: center; margin-bottom: 30px; }
    div[data-testid="stMetricValue"] { font-size: 28px !important; font-weight: bold; color: #1B5E20; }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# DOĞRUDAN VERİ YÜKLEME (ASLA CACHE YOK)
# ==========================================
csv_name = "SuperMarket Analysis_with_synthetic_features.csv"

# Streamlit'in önbellek çakışmalarını ezmek için doğrudan pandas okuması yapıyoruz
df = pd.read_csv(csv_name)
df['Date'] = pd.to_datetime(df['Date'])


# ==========================================
# ÜST BAŞLIK ALANI
# ==========================================
st.markdown("<div class='main-title'>EkoRaf Yönetim Paneli & Dijital İkiz Simülatörü</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Dış Değişken Duyarlı, Kademeli ve Otomatik Gıda Atığı Önleme Karar Destek Sistemi</div>", unsafe_allow_html=True)


# ==========================================
# SOL PANEL (SIDEBAR) FİLTRELERİ
# ==========================================
st.sidebar.header("📌 Mağaza Filtre Seçenekleri")
selected_city = st.sidebar.selectbox("Şehir Seçiniz", ["Tümü"] + list(df['City'].unique()))
selected_branch = st.sidebar.selectbox("Mağaza Şubesi", ["Tümü"] + list(df['Branch'].unique()))
selected_product_line = st.sidebar.selectbox("Ürün Grubu", ["Tümü"] + list(df['Product line'].unique()))

filtered_df = df.copy()
if selected_city != "Tümü":
    filtered_df = filtered_df[filtered_df['City'] == selected_city]
if selected_branch != "Tümü":
    filtered_df = filtered_df[filtered_df['Branch'] == selected_branch]
if selected_product_line != "Tümü":
    filtered_df = filtered_df[filtered_df['Product line'] == selected_product_line]


# ==========================================
# ANA SEKMELER (TABS)
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Mevcut Durum Raporu", "🧠 EkoRaf Öneri Motoru", "❄️ Dijital İkiz & Senaryo Simülatörü"])


# ------------------------------------------
# SEKME 1: MEVCUT DURUM VE SÜRDÜRÜLEBİLİRLİK
# ------------------------------------------
with tab1:
    st.subheader("🌿 Sürdürülebilirlik ve Finansal Göstergeler")
    
    saved_products = filtered_df[filtered_df['SKT_Kalan_Gun'] <= 10]
    total_saved_quantity = saved_products['Quantity'].sum()
    prevented_waste_tons = (total_saved_quantity * 0.4) / 1000 
    prevented_co2_tons = (prevented_waste_tons * 1000 * 2.5) / 1000
    total_gross_income = filtered_df['gross income'].sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="💰 Toplam Brüt Kâr", value=f"${total_gross_income:,.2f}")
    with col2:
        st.metric(label="🍏 Engellenen Gıda İsrafı", value=f"{prevented_waste_tons:.2f} Ton", delta="📉 İsraf Azaldı")
    with col3:
        st.metric(label="🌱 Engellenen Karbon Salınımı", value=f"{prevented_co2_tons:.2f} Ton $CO_2$e")
    with col4:
        st.metric(label="📦 Kurtarılan Ürün Adedi", value=f"{total_saved_quantity:,} Adet")

    st.markdown("---")
    st.subheader("📈 Kademeli İndirim ve Satış Davranış Analizi")
    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        avg_discount_by_skt = df.groupby('SKT_Kalan_Gun')['Uygulanan_Indirim_Orani'].mean().reset_index()
        fig1 = px.line(
            avg_discount_by_skt, x='SKT_Kalan_Gun', y='Uygulanan_Indirim_Orani',
            title='SKT Yaklaştıkça Kademeli Artan İndirim Oranı Ortalama Grafiği',
            labels={'SKT_Kalan_Gun': 'Son Tüketim Tarihine Kalan Gün', 'Uygulanan_Indirim_Orani': 'Ortalama İndirim (%)'},
            markers=True
        )
        fig1.update_traces(line_color='#2E7D32')
        st.plotly_chart(fig1, use_container_width=True)
        
    with g_col2:
        weather_sales = filtered_df.groupby('Hava_Durumu')['Quantity'].sum().reset_index()
        fig2 = px.bar(
            weather_sales, x='Hava_Durumu', y='Quantity',
            title='Dış Değişken Etkisi: Hava Durumuna Göre Satış Adetleri',
            labels={'Hava_Durumu': 'Hava Durumu', 'Quantity': 'Satılan Toplam Ürün (Adet)'},
            color='Hava_Durumu',
            color_discrete_sequence=px.colors.qualitative.Dark2
        )
        st.plotly_chart(fig2, use_container_width=True)



# ------------------------------------------
# SEKME 2: ANLIK REAKSİYON VE ÖNERİ MOTORU (DÜZENLENMİŞ SÜRÜM)
# ------------------------------------------
with tab2:
    st.subheader("🤖 EkoRaf Proaktif Aksiyon Dağıtım Ekranı")
    st.write("Sistem, ürünlerin son gününü beklemeden risk durumlarını analiz eder ve dış etkenlere göre anlık mikro-aksiyonlar üretir.")
    
    # Risk altındaki ürünleri filtrele
    risk_df = filtered_df[filtered_df['SKT_Kalan_Gun'] <= 5].head(10).copy()
    
    if not risk_df.empty:
        # Mahmut'un bahsettiği o uzun mesajı tabloyu daraltmak için çok daha kısa ve profesyonel aksiyon etiketlerine çeviriyoruz
        risk_df['EkoRaf_Aksiyonu'] = risk_df.apply(
            lambda row: "🚨 Mobil Fırsat + %" + f"{row['Uygulanan_Indirim_Orani']:.0f} İndirim"
            if row['Hava_Durumu'] in ['Yağmurlu', 'Karlı'] 
            else "📉 Kademeli Plan %" + f"{row['Uygulanan_Indirim_Orani']:.0f}", axis=1
        )
        
        # Gösterilebilecek en temiz sütunları seçiyoruz
        display_cols = ['Invoice ID', 'Product line', 'SKT_Kalan_Gun', 'Hava_Durumu', 'Uygulanan_Indirim_Orani', 'EkoRaf_Aksiyonu']
        
        # Sütunları Türkçeleştirip genişliklerini sabitliyoruz (Mahmut'un istediği düzenleme tam olarak burası)
        st.dataframe(
            risk_df[display_cols],
            use_container_width=True,
            hide_index=True, # Sol taraftaki gereksiz 0,1,2,3 sıra numaralarını gizler
            column_config={
                "Invoice ID": st.column_config.TextColumn("Fatura / Ürün ID", width="small"),
                "Product line": st.column_config.TextColumn("Ürün Kategorisi", width="medium"),
                "SKT_Kalan_Gun": st.column_config.NumberColumn("⏳ SKT Kalan", format="%d Gün", width="small"),
                "Hava_Durumu": st.column_config.TextColumn("🌤️ Hava", width="small"),
                "Uygulanan_Indirim_Orani": st.column_config.NumberColumn("📉 Önerilen İndirim", format="%%%d", width="small"),
                "EkoRaf_Aksiyonu": st.column_config.TextColumn("⚡ EkoRaf Sistem Aksiyonu", width="large")
            }
        )
    else:
        st.info("Harika! Seçilen kriterlere göre şu an kritik risk altında (SKT <= 5 Gün) ürün bulunmamaktadır.")
# ------------------------------------------
# SEKME 3: DİJİTAL İKİZ & SENARYO SİMÜLATÖRÜ
# ------------------------------------------
with tab3:
    st.subheader("🔮 Fiziksel Dünyanın Dijital İkiz Simülasyonu")
    st.write("Mağaza müdürü olarak gelecekteki olası bir dış etken senaryosunu seçip stoklarınızın erime hızını ve risk durumunu test edin.")
    
    sim_col1, sim_col2 = st.columns([1, 2])
    
    with sim_col1:
        st.markdown("#### 🛠️ Senaryo Parametreleri")
        scen_weather = st.selectbox("Hafta Sonu Beklenen Hava Durumu:", ["Karlı", "Yağmurlu", "Güneşli"])
        scen_product = st.selectbox("Simüle Edilecek Ürün Grubu:", df['Product line'].unique())
        scen_skt = st.slider("Ürünlerin Kalan Raf Ömrü (Gün):", 1, 10, 3)
        
        run_sim = st.button("🚀 Senaryoyu Dijital İkizde Çalıştır")
        
    with sim_col2:
        st.markdown("#### 📊 Simülasyon Çıktıları ve Tahminleme Modeli")
        
        if run_sim:
            base_speed = 100
            
            if scen_weather == "Karlı":
                drop_rate = 45 if scen_product == "Food and beverages" else 25
                eco_action = f"EkoRaf Kararı: Hafta sonu kar yağışı sebebiyle {scen_product} talebi düşecek! Çöpe gitmemesi için fiyatı hemen %15 ekstra aşağı çek ve mobil uygulama bildirimi tetikle."
                boost_factor = 1.35
            elif scen_weather == "Yağmurlu":
                drop_rate = 30 if scen_product == "Food and beverages" else 15
                eco_action = f"EkoRaf Kararı: Yağmurlu hava nedeniyle satış hızı düşüşü tespit edildi. Fiyatı anlık %12 düşür ve 'Yakın Mağaza Fırsatları' sekmesinde yayına al."
                boost_factor = 1.25
            else:
                drop_rate = 0
                eco_action = "EkoRaf Kararı: Hava koşulları ideal. Agresif ekstra aksiyona gerek yok, kademeli standart fiyat indirim planı uygulanıyor."
                boost_factor = 1.0
                
            simulated_velocity_without_ekoraf = base_speed - drop_rate
            simulated_velocity_with_ekoraf = simulated_velocity_without_ekoraf * boost_factor
            
            st.info(f"📋 **Senaryo Analizi:** Hafta sonu hava **{scen_weather}** olursa ve elinizde **{scen_skt} gün kalmış {scen_product}** stoku varsa:")
            
            v_col1, v_col2 = st.columns(2)
            v_col1.metric("EkoRaf Olmasaydı Stok Erime Hızı", f"%{simulated_velocity_without_ekoraf:.0f}", delta=f"-%{drop_rate} Düşüş", delta_color="inverse")
            v_col2.metric("EkoRaf ile Optimize Erime Hızı", f"%{simulated_velocity_with_ekoraf:.0f}", delta=f"+%{simulated_velocity_with_ekoraf - simulated_velocity_without_ekoraf:.0f} Kurtarma")
            
            categories = ['Normal Hız (%100)', 'EkoRaf Olmasa (Geleneksel)', 'EkoRaf Aktif (Akıllı İndirim)']
            velocities = [base_speed, simulated_velocity_without_ekoraf, simulated_velocity_with_ekoraf]
            
            fig_sim = go.Figure([go.Bar(x=categories, y=velocities, marker_color=['#90caf9', '#ef5350', '#66bb6a'])])
            fig_sim.update_layout(title="Stok Erime Hızı Karşılaştırma Grafiği (%)", yaxis_range=[0, 150])
            st.plotly_chart(fig_sim, use_container_width=True)
            
            st.success(f"⚡ {eco_action}")
            
            waste_prob_no_eko = max(0, 100 - simulated_velocity_without_ekoraf)
            waste_prob_with_eko = max(0, 100 - simulated_velocity_with_ekoraf)
            st.warning(f"🚨 **Çöpe Gitme / Bozulma Riski:** Geleneksel sistemde **%{waste_prob_no_eko:.0f}** iken, EkoRaf akıllı algoritmasıyla bu risk **%{waste_prob_with_eko:.0f}** seviyesine düşürülmüştür.")
        else:
            st.write("👈 Sol taraftaki parametreleri belirleyip **'Senaryoyu Dijital İkizde Çalıştır'** butonuna basarak fiziksel dünyayı simüle edebilirsiniz.")
