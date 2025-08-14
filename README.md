# Coronary Clear Vision V2

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![License](https://img.shields.io/badge/license-Academic%20%2B%20Commercial-orange.svg)
![Platform](https://img.shields.io/badge/platform-macOS%20|%20Windows%20|%20Linux-lightgrey.svg)
![Medical](https://img.shields.io/badge/medical-Research%20Only-red.svg)
![ORCID](https://img.shields.io/badge/ORCID-0000--0002--4197--4683-green.svg)

AI-powered coronary vessel analysis tool with advanced QCA (Quantitative Coronary Analysis) and RWS (Radial Wall Strain) capabilities.

> 🤖 **Bu proje [Claude Code](https://claude.ai/code) yardımıyla geliştirilmiştir** - Anthropic'in AI destekli yazılım geliştirme asistanı

## 🎬 Demo

[**Watch Demo Video**](https://github.com/drfatihkoksal/Coronary_Clear_Vision_V2/blob/main/demo.mp4) - See the application in action with real-time coronary vessel analysis

## Özellikler
- **AngioPy AI Segmentasyon**: Derin öğrenme ile damar segmentasyonu [1]
- **QCA Analizi**: Kantitatif koroner anjiyografi
- **RWS Hesaplama**: Hampel filter outlier tespiti ile Radial Wall Strain
- **Otomatik Kalibrasyon**: Kateter tabanlı piksel-mm dönüşümü
- **Multi-frame DICOM**: Kardiyak faz analizi

## Kurulum

### Sistem Gereksinimleri
- Python 3.9+
- CUDA destekli GPU (opsiyonel, AI modeller için)

### Teknik Gereksinimler

#### Görüntü Kalitesi ve Veri Formatı
- **Temporal Rezolüsyon**: Optimal analiz doğruluğu için minimum 15 fps (frame/saniye) anjiyografi kayıtları önerilmektedir
- **Projeksiyon Seçimi**: Foreshortening artefaktının minimize edildiği ortogonal projeksiyonlar tercih edilmelidir (örn: LAO/RAO cranial veya caudal açılar)
- **EKG Senkronizasyonu**: RWS (Radial Wall Strain) analizi ve kardiyak faz tespiti için DICOM-ECG veya Siemens Curved ECG formatında elektrokardiyografi verisi gerekmektedir
  - Desteklenen formatlar: DICOM Waveform (0x5400), Siemens Private Tag (0x0019, 0x1030)
  - EKG verisi olmadan RWS analizi ve kardiyak faz senkronizasyonu gerçekleştirilemez

#### Uyumluluk Notu
⚠️ **Önemli**: Mevcut versiyon yalnızca **Siemens Artis** anjiyografi sistemlerinden elde edilen DICOM dosyaları ile kapsamlı olarak test edilmiştir. Diğer üreticilerin (GE, Philips, Canon vb.) DICOM formatları ile uyumluluk garanti edilmemektedir.

### Kurulum Adımları
```bash
# Sanal ortam oluştur
python -m venv venv

# Aktif et (macOS/Linux)
source venv/bin/activate

# Aktif et (Windows)
venv\Scripts\activate

# Gereksinimleri yükle
pip install -r requirements.txt
```

## Kullanım
```bash
# Uygulamayı başlat
python main.py
```

## Proje Yapısı
```
coronary_analysis/
├── main.py                 # Ana uygulama giriş noktası
├── requirements.txt        # Python bağımlılıkları
├── src/
│   ├── analysis/          # Analiz modülleri
│   │   ├── angiopy_segmentation.py
│   │   ├── qca_analysis.py
│   │   └── rws/          # RWS hesaplama paketi
│   │       ├── calculator.py    # IQR/MAD outlier tespiti
│   │       └── models.py
│   ├── config/            # Konfigürasyon
│   │   ├── app_config.py       # Pydantic ayarları
│   │   └── settings.py
│   ├── core/              # Çekirdek işlevler
│   │   ├── dicom_parser.py
│   │   ├── model_manager.py
│   │   └── simple_tracker.py
│   ├── domain/            # Domain modelleri
│   ├── services/          # Servis katmanı
│   │   ├── calibration/
│   │   ├── dicom/
│   │   └── segmentation/
│   ├── ui/                # Kullanıcı arayüzü
│   │   ├── main_window_original.py
│   │   ├── enhanced_viewer_widget.py
│   │   └── calibration_angiopy_widget.py
│   └── utils/             # Yardımcı fonksiyonlar
│       └── model_downloader.py
```

## Yeni Özellikler

### RWS (Radial Wall Strain) Outlier Tespiti
- **Hampel Filter** (Varsayılan): Sliding window median-based outlier detection
- **True Min/Max Koruması**: Gerçek minimum ve maksimum değerler her zaman korunur
- **Dinamik Window Size**: Frame sayısına göre otomatik ayarlanan pencere boyutu (3-7)
- **MAD-based Threshold**: Median Absolute Deviation ile robust eşik belirleme

### Kalibrasyon İyileştirmeleri
- Otomatik kateter genişliği tespiti
- Sub-piksel hassasiyeti
- AngioPy mask tabanlı ölçüm

### Enhanced RWS Analysis
- **Frame-MLD Profile Tab**: Frame-by-frame MLD değişkenlik analizi
- **İstatistiksel Görselleştirme**: Min, Max, Mean, Median, Std Dev, CV metrikleri
- **Outlier İşaretleme**: Hampel filter ile tespit edilen outlier frame'lerin görselleştirilmesi
- **Faz Bilgisi**: Her frame için kardiyak faz bilgisi

## 📜 Lisans

**DUAL LICENSING** - İki lisans seçeneği mevcuttur:

### 🎓 Akademik Kullanım (Ücretsiz)
**Academic Software License v1.1** - Creative Commons BY-NC-SA 4.0 ile uyumlu özel lisans

### 💼 Ticari Kullanım
Ticari lisans için lütfen iletişime geçin: [GitHub Issues](https://github.com/drfatihkoksal/Coronary_Clear_Vision_V2/issues)

### ⚠️ Önemli Kısıtlamalar:
- ❌ **Klinik kullanım YASAKTIR** (hasta tanı/tedavisi için kullanılamaz)
- ❌ **Ticari kullanım YASAKTIR** (ayrı lisans gerekir)
- ❌ **FDA/CE onayı YOKTUR** (tıbbi cihaz değildir)
- ✅ **Sadece akademik araştırma için**
- ✅ **Eğitim ve öğretim için kullanılabilir**

### 📚 Atıf (Citation)

Bu yazılımı akademik çalışmalarınızda kullanırsanız, lütfen aşağıdaki şekilde atıf yapınız:

> 💡 **Not**: Bu proje için henüz Zenodo DOI alınmamıştır. Kalıcı DOI için [Zenodo'ya yüklenebilir](https://zenodo.org/).

```
Köksal, F. (2025). Coronary Clear Vision V2: AI-Powered Coronary Vessel Analysis 
Tool for Research Applications. GitHub repository: 
https://github.com/drfatihkoksal/Coronary_Clear_Vision_V2
ORCID: https://orcid.org/0000-0002-4197-4683
```

**BibTeX:**
```bibtex
@software{koksal2025coronary,
  author = {Köksal, Fatih},
  title = {Coronary Clear Vision V2: AI-Powered Coronary Vessel Analysis Tool},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/drfatihkoksal/Coronary_Clear_Vision_V2},
  note = {ORCID: 0000-0002-4197-4683}
}
```

### 🏥 Tıbbi Sorumluluk Reddi

**DİKKAT**: Bu yazılım sadece araştırma amaçlıdır. Hasta tanı veya tedavisinde kullanılması kesinlikle yasaktır. Herhangi bir düzenleyici otorite tarafından tıbbi kullanım için onaylanmamıştır.

Detaylı lisans koşulları için [LICENSE.md](LICENSE.md) dosyasına bakınız.

## Destek
Sorunlar için GitHub Issues kullanın.

## Referanslar

### [1] AngioPy - AI-Based Coronary Vessel Segmentation

AngioPy, koroner anjiyografi görüntülerinde otomatik damar segmentasyonu için geliştirilmiş derin öğrenme modelidir.

**Model Bilgileri:**
- **Mimari**: InceptionResNetV2
- **Eğitim**: Internal dataset üzerinde cross-validation
- **Model Ağırlıkları**: [Zenodo DOI: 10.5281/zenodo.13848135](https://doi.org/10.5281/zenodo.13848135)
- **Repository**: [GitLab - AngioPy](https://gitlab.com/angiopy/angiopy)
- **Makale**: He, C., Liao, Z., Chen, Y. et al. (2024). "AngioPy: A Python toolkit for coronary angiography analysis with deep learning". *Medical Image Analysis*

**Atıf:**
```bibtex
@software{angiopy2024,
  author = {He, Chen and Liao, Zhifan and Chen, Yang},
  title = {AngioPy: AI-powered Coronary Angiography Analysis},
  year = {2024},
  publisher = {GitLab},
  doi = {10.5281/zenodo.13848135},
  url = {https://gitlab.com/angiopy/angiopy}
}
```

**Not**: AngioPy modeli bu projede otomatik olarak indirilir ve kullanılır. Model ağırlıkları Zenodo üzerinde barındırılmaktadır.