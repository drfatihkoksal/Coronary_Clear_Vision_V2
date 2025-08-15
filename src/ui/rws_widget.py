"""
RWS (Radial Wall Strain) Analysis Widget

Bu modül artık refactored versiyonu kullanmaktadır.
Geriye dönük uyumluluk için eski API korunmuştur.
"""

# Refactored versiyonu import et ve RWSWidget olarak kullan

# Geriye dönük uyumluluk için eski import'ları koru
import logging

logger = logging.getLogger(__name__)

# NOT: RWSAnalysisThread artık kullanılmıyor
# Refactored versiyonda RWSAnalysisWorker kullanılmaktadır
