import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import glob
import os

warnings.filterwarnings('ignore')

# ============================================
# STYLING VE KONFIGÜRASYON
# ============================================
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
RANDOM_STATE = 42

# ============================================
# LOGGER SETUP
# ============================================
class ColorLogger:
    """Renkli ve şık log çıktısı için sınıf"""
    
    COLORS = {
        'HEADER': '\033[95m',
        'BLUE': '\033[94m',
        'CYAN': '\033[96m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'RED': '\033[91m',
        'BOLD': '\033[1m',
        'UNDERLINE': '\033[4m',
        'END': '\033[0m'
    }
    
    @staticmethod
    def header(text):
        print(f"\n{ColorLogger.COLORS['BOLD']}{ColorLogger.COLORS['CYAN']}{'='*80}")
        print(f"  {text}")
        print(f"{'='*80}{ColorLogger.COLORS['END']}\n")
    
    @staticmethod
    def success(text):
        print(f"{ColorLogger.COLORS['GREEN']}✓ {text}{ColorLogger.COLORS['END']}")
    
    @staticmethod
    def info(text):
        print(f"{ColorLogger.COLORS['BLUE']}ℹ {text}{ColorLogger.COLORS['END']}")
    
    @staticmethod
    def warning(text):
        print(f"{ColorLogger.COLORS['YELLOW']}⚠ {text}{ColorLogger.COLORS['END']}")
    
    @staticmethod
    def error(text):
        print(f"{ColorLogger.COLORS['RED']}✗ {text}{ColorLogger.COLORS['END']}")
    
    @staticmethod
    def section(text):
        print(f"\n{ColorLogger.COLORS['BOLD']}{ColorLogger.COLORS['YELLOW']}{text}{ColorLogger.COLORS['END']}")

logger = ColorLogger()

# ============================================
# 1. VERİ YÜKLEME (OTOMATİK DOSYA BULMA)
# ============================================
logger.header("🚀 EkoRaf - XGBoost Model Eğitim Pipeline v2.0")
logger.section("📂 Adım 1: Veri Yükleme")

# CSV dosyasını otomatik bul
csv_files = glob.glob('*synthetic_features.csv')

if not csv_files:
    logger.error("*synthetic_features.csv dosyası bulunamadı!")
    logger.info("Lütfen şu dosyalardan birini çalıştır:")
    logger.info("  1. supermarket_sales_with_synthetic_features.csv")
    logger.info("  2. SuperMarket Analysis_with_synthetic_features.csv")
    exit(1)

csv_file = csv_files[0]
logger.success(f"CSV dosyası bulundu: {csv_file}")

try:
    df = pd.read_csv(csv_file)
    logger.success(f"Veri seti yüklendi: {df.shape[0]} satır × {df.shape[1]} sütun")
    logger.info(f"Veri seti boyutu: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
except FileNotFoundError:
    logger.error(f"{csv_file} bulunamadı!")
    exit(1)

# Veri kalitesi kontrol
logger.info(f"Eksik değerler: {df.isnull().sum().sum()}")
logger.info(f"Sütunlar: {list(df.columns)}")

# ============================================
# 2. SENTETİK KOLONLAR KONTROL VE OLUŞTURMA
# ============================================
logger.section("🔄 Adım 2: Sentetik Kolonlar Kontrol/Oluşturma")

required_columns = ['SKT_Kalan_Gun', 'Hava_Durumu', 'Uygulanan_Indirim_Orani']

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    logger.warning(f"Eksik kolonlar tespit edildi: {missing_columns}")
    logger.info("Sentetik kolonlar oluşturuluyor...")
    
    # Eğer eksikse oluştur
    if 'SKT_Kalan_Gun' not in df.columns:
        skt_values = []
        for product_line in df['Product line']:
            if product_line == 'Food and beverages':
                if np.random.random() < 0.8:
                    skt_values.append(np.random.randint(1, 8))
                else:
                    skt_values.append(np.random.randint(8, 31))
            else:
                skt_values.append(np.random.randint(1, 31))
        df['SKT_Kalan_Gun'] = skt_values
        logger.success("SKT_Kalan_Gun kolonu oluşturuldu")
    
    if 'Hava_Durumu' not in df.columns:
        df['Hava_Durumu'] = np.random.choice(['Güneşli', 'Yağmurlu', 'Bulutlu', 'Karlı'], len(df))
        logger.success("Hava_Durumu kolonu oluşturuldu")
    
    if 'Uygulanan_Indirim_Orani' not in df.columns:
        indirim_values = []
        for skt_gun in df['SKT_Kalan_Gun']:
            base_discount = (30 - skt_gun) * 1.67
            noise = np.random.uniform(-5, 5)
            discount = max(0, min(50, base_discount + noise))
            indirim_values.append(round(discount, 2))
        df['Uygulanan_Indirim_Orani'] = indirim_values
        logger.success("Uygulanan_Indirim_Orani kolonu oluşturuldu")
else:
    logger.success("Tüm gerekli sentetik kolonlar mevcut!")

# ============================================
# 3. ÖZELLİK SEÇİMİ VE HAZIRLIK
# ============================================
logger.section("🔍 Adım 3: Özellik Seçimi ve Ön İşleme")

feature_columns = ['SKT_Kalan_Gun', 'Hava_Durumu', 'Product line', 'Unit price']
target_column = 'Uygulanan_Indirim_Orani'

X = df[feature_columns].copy()
y = df[target_column].copy()

logger.success(f"Özellikler seçildi: {feature_columns}")
logger.success(f"Hedef değişken: {target_column}")

# Hedef değişken istatistikleri
logger.info(f"Hedef değişken istatistikleri:")
print(f"  Min: {y.min():.2f}% | Max: {y.max():.2f}% | Ortalama: {y.mean():.2f}% | Std: {y.std():.2f}%")

# ============================================
# 4. VERİ ÖN İŞLEME (ONE-HOT ENCODING)
# ============================================
logger.section("⚙️  Adım 4: Kategorik Özelliklerin Kodlanması")

categorical_features = ['Hava_Durumu', 'Product line']
logger.info(f"Kategorik özellikler: {categorical_features}")

# One-Hot Encoding
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=False)

logger.success(f"One-Hot Encoding tamamlandı")
logger.info(f"Özellik sayısı: {X.shape[1]} → {X_encoded.shape[1]}")

# Sütun isimlerini sakla (API'de kullanılacak)
model_columns = X_encoded.columns.tolist()
logger.info(f"Kodlanmış özellikler (ilk 10): {model_columns[:10]}")

# ============================================
# 5. EĞİTİM/TEST AYRIMI
# ============================================
logger.section("📊 Adım 5: Eğitim/Test Ayrımı (%80/%20)")

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, 
    test_size=0.2, 
    random_state=RANDOM_STATE
)

logger.success(f"Eğitim seti: {X_train.shape[0]} satır ({(X_train.shape[0]/len(X))*100:.1f}%)")
logger.success(f"Test seti: {X_test.shape[0]} satır ({(X_test.shape[0]/len(X))*100:.1f}%)")

# ============================================
# 6. MODEL OLUŞTURMA VE EĞİTİMİ
# ============================================
logger.section("🤖 Adım 6: XGBoost Modeli Eğitimi")

# Optimizasyon hiperparametreleri
hyperparams = {
    'n_estimators': 150,
    'max_depth': 6,
    'learning_rate': 0.08,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'gamma': 1,
    'min_child_weight': 3,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbosity': 0
}

logger.info(f"Hiperparametreler:")
for param, value in hyperparams.items():
    print(f"  • {param}: {value}")

model = xgb.XGBRegressor(**hyperparams)

# Modeli eğit
logger.info("Model eğitiliyor... (Bu işlem biraz sürebilir)")
model.fit(X_train, y_train, verbose=False)

logger.success("Model eğitimi tamamlandı!")

# ============================================
# 7. CROSS-VALIDATION (K-FOLD)
# ============================================
logger.section("🔄 Adım 7: Cross-Validation (5-Fold)")

cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
logger.success(f"Cross-Validation R² Skorları: {[f'{score:.4f}' for score in cv_scores]}")
logger.info(f"Ortalama R² (CV): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================
# 8. MODEL DEĞERLENDİRMESİ
# ============================================
logger.section("📈 Adım 8: Model Performansı")

# Tahminler
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Metrikler
metrics = {
    'train': {
        'MAE': mean_absolute_error(y_train, y_train_pred),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'MAPE': mean_absolute_percentage_error(y_train, y_train_pred),
        'R2': r2_score(y_train, y_train_pred)
    },
    'test': {
        'MAE': mean_absolute_error(y_test, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAPE': mean_absolute_percentage_error(y_test, y_test_pred),
        'R2': r2_score(y_test, y_test_pred)
    }
}

# Profesyonel tablo gösterimi
print(f"\n{'Metrik':<15} {'Eğitim Seti':<20} {'Test Seti':<20}")
print("-" * 55)
print(f"{'MAE (%):':<15} {metrics['train']['MAE']:<20.4f} {metrics['test']['MAE']:<20.4f}")
print(f"{'RMSE (%):':<15} {metrics['train']['RMSE']:<20.4f} {metrics['test']['RMSE']:<20.4f}")
print(f"{'MAPE (%):':<15} {metrics['train']['MAPE']:<20.4f} {metrics['test']['MAPE']:<20.4f}")
print(f"{'R² Skoru:':<15} {metrics['train']['R2']:<20.4f} {metrics['test']['R2']:<20.4f}")
print("-" * 55)

# Model sağlığı kontrolü
overfitting_gap = metrics['train']['R2'] - metrics['test']['R2']
if overfitting_gap > 0.15:
    logger.warning(f"Potansiyel overfitting (R² farkı: {overfitting_gap:.4f})")
else:
    logger.success(f"Model sağlığı iyi (R² farkı: {overfitting_gap:.4f})")

# ============================================
# 9. ÖZELLİK ÖNEMLİLİĞİ ANALİZİ
# ============================================
logger.section("🔥 Adım 9: Özellik Önemlilik Analizi")

feature_importance = pd.DataFrame({
    'Özellik': X_encoded.columns,
    'Önemlilik': model.feature_importances_
}).sort_values('Önemlilik', ascending=False)

logger.success("Top 10 Önemli Özellikler:")
print()
for idx, row in feature_importance.head(10).iterrows():
    bar_length = int(row['Önemlilik'] * 50)
    bar = "█" * bar_length
    print(f"  {row['Özellik']:<30} {bar} {row['Önemlilik']:.4f}")

# Özellik önemlilik grafiği
fig, ax = plt.subplots(figsize=(12, 6))
top_features = feature_importance.head(10)
colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
ax.barh(range(len(top_features)), top_features['Önemlilik'].values, color=colors)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Özellik'].values)
ax.set_xlabel('Önemlilik Skoru', fontsize=12, fontweight='bold')
ax.set_title('EkoRaf - Özellik Önemlilik Analizi', fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
logger.success("Özellik önemlilik grafiği kaydedildi: feature_importance.png")
plt.close()

# ============================================
# 10. TAHMIN vs GERÇEK GRAFİĞİ
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Eğitim seti
axes[0].scatter(y_train, y_train_pred, alpha=0.6, s=30, color='blue', edgecolors='navy')
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0].set_xlabel('Gerçek İndirim Oranı (%)', fontweight='bold')
axes[0].set_ylabel('Tahmin Edilen İndirim Oranı (%)', fontweight='bold')
axes[0].set_title(f'Eğitim Seti (R² = {metrics["train"]["R2"]:.4f})', fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Test seti
axes[1].scatter(y_test, y_test_pred, alpha=0.6, s=30, color='green', edgecolors='darkgreen')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Gerçek İndirim Oranı (%)', fontweight='bold')
axes[1].set_ylabel('Tahmin Edilen İndirim Oranı (%)', fontweight='bold')
axes[1].set_title(f'Test Seti (R² = {metrics["test"]["R2"]:.4f})', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
logger.success("Tahmin analizi grafiği kaydedildi: prediction_analysis.png")
plt.close()

# ============================================
# 11. HATA DAĞILIMI
# ============================================
test_errors = y_test - y_test_pred

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Hata histogramu
axes[0].hist(test_errors, bins=30, color='coral', edgecolor='black', alpha=0.7)
axes[0].axvline(test_errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Ort: {test_errors.mean():.2f}')
axes[0].set_xlabel('Hata (%) - Gerçek - Tahmin', fontweight='bold')
axes[0].set_ylabel('Frekans', fontweight='bold')
axes[0].set_title('Hata Dağılımı (Test Seti)', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Hata scatter
axes[1].scatter(y_test_pred, test_errors, alpha=0.6, s=30, color='purple', edgecolors='darkviolet')
axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Tahmin Edilen İndirim Oranı (%)', fontweight='bold')
axes[1].set_ylabel('Hata (%)', fontweight='bold')
axes[1].set_title('Hata Analizi', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
logger.success("Hata analizi grafiği kaydedildi: error_analysis.png")
plt.close()

# ============================================
# 12. ÖRNEK TAHMINLER (ÇETİN DURUMLAR)
# ============================================
logger.section("🧪 Adım 10: Örnek Tahminler")

# Hata sırasına göre en zor tahminleri bul
test_errors_abs = np.abs(test_errors)
difficult_indices = test_errors_abs.argsort()[-5:][::-1]

print(f"\n{'Sıra':<5} {'Gerçek':<12} {'Tahmin':<12} {'Hata':<12} {'% Hata':<12}")
print("-" * 53)

for idx, sample_idx in enumerate(difficult_indices, 1):
    actual = y_test.iloc[sample_idx]
    predicted = y_test_pred[sample_idx]
    error = actual - predicted
    error_pct = (abs(error) / actual * 100) if actual != 0 else 0
    print(f"{idx:<5} {actual:<12.2f}% {predicted:<12.2f}% {error:<12.2f}% {error_pct:<12.2f}%")

# ============================================
# 13. KONFIGÜRASYON DOSYASI OLUŞTUR
# ============================================
logger.section("📋 Adım 11: Model Metadata ve Konfigürasyon")

model_metadata = {
    'model_name': 'EkoRaf-XGBoost-v2.0',
    'creation_date': datetime.now().isoformat(),
    'model_type': 'XGBRegressor',
    'target_variable': target_column,
    'features': feature_columns,
    'training_samples': int(X_train.shape[0]),
    'test_samples': int(X_test.shape[0]),
    'total_features': int(X_encoded.shape[1]),
    'hyperparameters': hyperparams,
    'performance_metrics': {
        'train': {k: float(v) for k, v in metrics['train'].items()},
        'test': {k: float(v) for k, v in metrics['test'].items()}
    },
    'cross_validation_r2': {
        'mean': float(cv_scores.mean()),
        'std': float(cv_scores.std()),
        'scores': [float(s) for s in cv_scores]
    },
    'feature_importance': feature_importance.set_index('Özellik')['Önemlilik'].to_dict()
}

with open('ekoraf_model_config.json', 'w', encoding='utf-8') as f:
    json.dump(model_metadata, f, indent=2, ensure_ascii=False)

logger.success("Model metadata kaydedildi: ekoraf_model_config.json")

# ============================================
# 14. MODEL VE ARTIFACTS KAYDET
# ============================================
logger.section("💾 Adım 12: Model ve Artifacts Kaydetme")

# Model kaydet
joblib.dump(model, 'ekoraf_model.pkl')
logger.success("Model kaydedildi: ekoraf_model.pkl")

# Sütun isimlerini kaydet
joblib.dump(model_columns, 'model_columns.pkl')
logger.success("Model sütunları kaydedildi: model_columns.pkl")

# Scaler (isteğe bağlı ileri versionlar için)
scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, 'feature_scaler.pkl')
logger.success("Feature scaler kaydedildi: feature_scaler.pkl")

# ============================================
# 15. MODEL ÖZET RAPORU
# ============================================
logger.header("📊 FINAL ÖZET RAPORU")

summary_text = f"""
╔════════════════════════════════════════════════════════════════╗
║                  EkoRaf Model Training Summary                 ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  📌 MODEL DETAYLARI                                            ║
║  ├─ Model Türü: XGBoost Regressor                             ║
║  ├─ Sürüm: 2.0 (Profesyonel)                                  ║
║  ├─ Eğitim Tarihi: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}          ║
║  └─ Veri Kaynağı: {csv_file}                                  ║
║                                                                ║
║  📊 VERİ DÜZENİ                                                ║
║  ├─ Toplam Örnek: {len(X):,}                                  ║
║  ├─ Eğitim: {X_train.shape[0]:,} (%80)                         ║
║  ├─ Test: {X_test.shape[0]:,} (%20)                           ║
║  └─ Toplam Özellik: {X_encoded.shape[1]}                      ║
║                                                                ║
║  🎯 PERFORMANS (Test Seti)                                     ║
║  ├─ R² Skoru: {metrics['test']['R2']:.4f}                     ║
║  ├─ MAE: {metrics['test']['MAE']:.4f}%                        ║
║  ├─ RMSE: {metrics['test']['RMSE']:.4f}%                      ║
║  └─ MAPE: {metrics['test']['MAPE']:.4f}%                      ║
║                                                                ║
║  ✅ MODELIN SAĞLIĞI                                            ║
║  ├─ Overfit Riski: {"✓ Normal" if overfitting_gap <= 0.15 else "⚠ Yüksek"}                        ║
║  ├─ CV R² (Ort): {cv_scores.mean():.4f}                       ║
║  ├─ CV Std Dev: {cv_scores.std():.4f}                         ║
║  └─ Stabilite: {"✓ İyi" if cv_scores.std() < 0.05 else "⚠ Gözlemlenecek"}                  ║
║                                                                ║
║  📁 KAYDEDILEN DOSYALAR                                        ║
║  ├─ ekoraf_model.pkl                                          ║
║  ├─ model_columns.pkl                                         ║
║  ├─ feature_scaler.pkl                                        ║
║  ├─ ekoraf_model_config.json                                  ║
║  ├─ feature_importance.png                                    ║
║  ├─ prediction_analysis.png                                   ║
║  └─ error_analysis.png                                        ║
║                                                                ║
║  🚀 SONRAKI ADIMLAR                                            ║
║  ├─ 1. Model doğruluğunun üretime uygunluğunu değerlendir    ║
║  ├─ 2. FastAPI ile REST API oluştur                          ║
║  ├─ 3. Frontend dashboard bağla                              ║
║  └─ 4. A/B testing başlat                                    ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
"""

print(summary_text)

logger.success("Model eğitimi ve değerlendirmesi başarıyla tamamlandı!")
logger.info("Tüm dosyalar geçerli dizine kaydedildi.")
print("\n")