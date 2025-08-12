"""
Damar çapı ölçüm yöntemi seçici
Farklı ölçüm yöntemleri arasında kolayca geçiş yapmayı sağlar
"""

import logging

logger = logging.getLogger(__name__)


class DiameterMethodSelector:
    """Çap ölçüm yöntemi seçici"""
    
    AVAILABLE_METHODS = {
        'bifurcation_aware': 'Çatallanma-farkında hibrit yöntem (önerilen)',
        'centerline': 'Centerline tabanlı edge detection',
        'hybrid': 'Segmentasyon + centerline hibrit',
        'simple': 'Basit perpendicular yöntem',
        'advanced': 'Gelişmiş algoritma',
        'matlab': 'MATLAB-inspired yöntem'
    }
    
    @staticmethod
    def set_method(qca_analyzer, method_name: str):
        """
        QCA analizör için çap ölçüm yöntemini ayarla
        
        Args:
            qca_analyzer: QCAAnalysis instance
            method_name: Yöntem adı
        """
        # Tüm yöntemleri kapat
        qca_analyzer.use_bifurcation_aware = False
        qca_analyzer.use_centerline_diameter = False
        qca_analyzer.use_hybrid_diameter = False
        qca_analyzer.use_advanced_diameter = False
        qca_analyzer.use_matlab_diameter = False
        
        # Seçilen yöntemi aç
        if method_name == 'bifurcation_aware':
            qca_analyzer.use_bifurcation_aware = True
            logger.info("Çatallanma-farkında hibrit yöntem seçildi")
        elif method_name == 'centerline':
            qca_analyzer.use_centerline_diameter = True
            logger.info("Centerline tabanlı yöntem seçildi")
        elif method_name == 'hybrid':
            qca_analyzer.use_hybrid_diameter = True
            logger.info("Hibrit yöntem seçildi")
        elif method_name == 'advanced':
            qca_analyzer.use_advanced_diameter = True
            logger.info("Gelişmiş yöntem seçildi")
        elif method_name == 'matlab':
            qca_analyzer.use_matlab_diameter = True
            logger.info("MATLAB-inspired yöntem seçildi")
        else:
            # Varsayılan: bifurcation_aware
            qca_analyzer.use_bifurcation_aware = True
            logger.warning(f"Bilinmeyen yöntem: {method_name}. Varsayılan bifurcation_aware kullanılıyor.")
    
    @staticmethod
    def get_current_method(qca_analyzer) -> str:
        """Aktif çap ölçüm yöntemini öğren"""
        if qca_analyzer.use_bifurcation_aware:
            return 'bifurcation_aware'
        elif qca_analyzer.use_centerline_diameter:
            return 'centerline'
        elif qca_analyzer.use_hybrid_diameter:
            return 'hybrid'
        elif qca_analyzer.use_advanced_diameter:
            return 'advanced'
        elif qca_analyzer.use_matlab_diameter:
            return 'matlab'
        else:
            return 'simple'


def select_diameter_method(qca_analyzer, method='bifurcation_aware'):
    """
    Kısa yol fonksiyonu
    
    Kullanım:
        from src.analysis.diameter_method_selector import select_diameter_method
        select_diameter_method(qca_analyzer, 'bifurcation_aware')
    """
    DiameterMethodSelector.set_method(qca_analyzer, method)