"""
RWS (Radial Wall Strain) Analysis Widget

Bu modül artık refactored versiyonu kullanmaktadır.
Geriye dönük uyumluluk için eski API korunmuştur.
"""

# Refactored versiyonu import et ve RWSWidget olarak kullan
from src.ui.rws_widget_refactored import RWSWidgetRefactored as RWSWidget

# Geriye dönük uyumluluk için eski import'ları koru
from PyQt6.QtCore import pyqtSignal
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# NOT: RWSAnalysisThread artık kullanılmıyor
# Refactored versiyonda RWSAnalysisWorker kullanılmaktadır