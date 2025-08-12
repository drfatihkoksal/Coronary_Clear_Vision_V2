"""
Tracking tabanlı co-registration'ı etkinleştirmek için yardımcı modül
"""

def enable_tracking_coregistration(rws_analyzer):
    """
    RWS analizinde tracking tabanlı co-registration'ı etkinleştir
    
    Args:
        rws_analyzer: EnhancedRWSAnalysis instance
    """
    rws_analyzer.use_tracking_coregistration = True
    print("Tracking tabanlı co-registration ETKİN")
    
def disable_tracking_coregistration(rws_analyzer):
    """
    RWS analizinde tracking tabanlı co-registration'ı devre dışı bırak
    
    Args:
        rws_analyzer: EnhancedRWSAnalysis instance
    """
    rws_analyzer.use_tracking_coregistration = False
    print("Tracking tabanlı co-registration DEVRE DIŞI - Eski yöntem kullanılacak")
    
def toggle_tracking_coregistration(rws_analyzer):
    """
    Tracking tabanlı co-registration durumunu değiştir
    
    Args:
        rws_analyzer: EnhancedRWSAnalysis instance
    """
    rws_analyzer.use_tracking_coregistration = not rws_analyzer.use_tracking_coregistration
    status = "ETKİN" if rws_analyzer.use_tracking_coregistration else "DEVRE DIŞI"
    print(f"Tracking tabanlı co-registration {status}")
    
def get_coregistration_status(rws_analyzer):
    """
    Tracking tabanlı co-registration durumunu öğren
    
    Args:
        rws_analyzer: EnhancedRWSAnalysis instance
        
    Returns:
        bool: True if enabled, False otherwise
    """
    return getattr(rws_analyzer, 'use_tracking_coregistration', False)