# Python Proje Çalıştırma Rehberi

## ÖNEMLİ: Sanal Ortam Kullanımı

Bu proje Python sanal ortamı (venv) kullanmaktadır.

### MUTLAKA İZLENECEK ADIMLAR:

#### main.py çalıştırmadan önce:
1. **Önce sanal ortamı aktif et:**
   ```bash
   source venv/bin/activate
   ```

2. **Sonra Python scriptini çalıştır:**
   ```bash
   python main.py
   ```

### Sanal Ortam Bilgileri:
- **Venv konumu:** `./venv/`
- **Aktivasyon komutu:** `source venv/bin/activate`
- **Deaktivasyon:** `deactivate`

### KURAL: 
**HER PYTHON KOMUTU ÖNCESİNDE VENV AKTİF OLMALI!**

#### Örnek kullanım:
```bash
# Doğru kullanım
source venv/bin/activate
python main.py

# Yanlış kullanım  
python main.py  # Venv aktif değil!
```

### Paket kurulumu da venv içinde yapılmalı:
```bash
source venv/bin/activate
pip install paket_adi
```

### Gereksinimler dosyası:
```bash
# Mevcut paketleri kaydetmek için
source venv/bin/activate
pip freeze > requirements.txt

# Paketleri yüklemek için
source venv/bin/activate
pip install -r requirements.txt
```

### Yaygın Komutlar:
```bash
# Venv oluşturma (ilk kurulum)
python -m venv venv

# Venv aktif etme
source venv/bin/activate

# Python script çalıştırma
python main.py

# Venv deaktif etme
deactivate
```

## Proje Hakkında
Bu dosya Claude Code'a projenin nasıl çalıştırılacağını öğretir. Claude bu talimatları her proje açışında okur ve uygular.
