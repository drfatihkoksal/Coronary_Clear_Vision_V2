"""
Motion Prediction Implementations

Hareket tahmini için farklı algoritmalar.
Kalp döngüsü ve damar hareketi için özelleştirilmiş.
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy import signal
import logging
from abc import ABC

from src.domain.models.tracking_models import Point2D
from src.domain.interfaces.tracking_interfaces import IMotionPredictor

logger = logging.getLogger(__name__)


class BaseMotionPredictor(ABC):
    """
    Hareket tahmini base class.

    Ortak fonksiyonalite ve yardımcı metodlar.
    """

    def __init__(self, history_size: int = 10, fps: float = 30.0):
        """
        BaseMotionPredictor constructor.

        Args:
            history_size: Geçmiş nokta sayısı
            fps: Video frame rate (tahmin için önemli)
        """
        self.history_size = history_size
        self.fps = fps
        self.frame_interval = 1.0 / max(fps, 1.0)  # Saniye cinsinden frame aralığı
        self.position_history: List[Point2D] = []
        self.velocity_history: List[Tuple[float, float]] = []
        self.acceleration_history: List[Tuple[float, float]] = []

    def _update_history(self, position: Point2D):
        """
        Geçmiş verisini güncelle.

        Args:
            position: Yeni pozisyon
        """
        # Pozisyon geçmişi
        self.position_history.append(position)
        if len(self.position_history) > self.history_size:
            self.position_history.pop(0)

        # Hız hesapla
        if len(self.position_history) >= 2:
            prev = self.position_history[-2]
            curr = self.position_history[-1]
            velocity = (curr.x - prev.x, curr.y - prev.y)
            self.velocity_history.append(velocity)

            if len(self.velocity_history) > self.history_size:
                self.velocity_history.pop(0)

        # İvme hesapla
        if len(self.velocity_history) >= 2:
            prev_vel = self.velocity_history[-2]
            curr_vel = self.velocity_history[-1]
            acceleration = (curr_vel[0] - prev_vel[0], curr_vel[1] - prev_vel[1])
            self.acceleration_history.append(acceleration)

            if len(self.acceleration_history) > self.history_size:
                self.acceleration_history.pop(0)

    def _smooth_trajectory(self, positions: List[Point2D], window_size: int = 3) -> List[Point2D]:
        """
        Hareket yörüngesini yumuşat.

        Args:
            positions: Pozisyon listesi
            window_size: Yumuşatma pencere boyutu

        Returns:
            List[Point2D]: Yumuşatılmış pozisyonlar
        """
        if len(positions) < window_size:
            return positions

        # X ve Y koordinatlarını ayır
        x_coords = [p.x for p in positions]
        y_coords = [p.y for p in positions]

        # Hareketli ortalama
        kernel = np.ones(window_size) / window_size
        x_smooth = np.convolve(x_coords, kernel, mode="valid")
        y_smooth = np.convolve(y_coords, kernel, mode="valid")

        # Yumuşatılmış pozisyonlar
        smoothed = []

        # Başlangıç noktaları (yumuşatılamayan)
        pad_size = window_size // 2
        for i in range(pad_size):
            smoothed.append(positions[i])

        # Yumuşatılmış noktalar
        for i in range(len(x_smooth)):
            smoothed.append(Point2D(x_smooth[i], y_smooth[i]))

        # Bitiş noktaları (yumuşatılamayan)
        for i in range(len(positions) - len(smoothed)):
            smoothed.append(positions[-(i + 1)])

        return smoothed


class LinearMotionPredictor(BaseMotionPredictor, IMotionPredictor):
    """
    Doğrusal hareket tahmini.

    Sabit hız varsayımı ile tahmin yapar.
    Basit ve hızlı, düz hareketler için uygun.
    FPS-aware: Düşük FPS için daha agresif tahmin.
    """
    
    def __init__(self, history_size: int = 10, fps: float = 30.0):
        """
        LinearMotionPredictor constructor.
        
        Args:
            history_size: Geçmiş nokta sayısı
            fps: Video frame rate
        """
        super().__init__(history_size, fps)

    def predict(self, history: List[Point2D], steps_ahead: int = 1) -> List[Point2D]:
        """
        Gelecek pozisyonları tahmin eder (FPS-aware).

        Args:
            history: Geçmiş pozisyonlar
            steps_ahead: Kaç adım ileri tahmin

        Returns:
            List[Point2D]: Tahmin edilen pozisyonlar
        """
        if len(history) < 2:
            # Yetersiz veri, son noktayı tekrarla
            if history:
                return [history[-1]] * steps_ahead
            else:
                return [Point2D(0, 0)] * steps_ahead

        # Düşük FPS için daha fazla geçmiş noktayı kullan
        if self.fps < 20 and len(history) >= 3:
            # Son 3 noktadan ortalama hız hesapla
            velocities_x = []
            velocities_y = []
            
            for i in range(1, min(4, len(history))):
                vx = history[-i].x - history[-(i+1)].x
                vy = history[-i].y - history[-(i+1)].y
                velocities_x.append(vx)
                velocities_y.append(vy)
            
            # Ağırlıklı ortalama (yakın zamana daha fazla ağırlık)
            weights = [0.5, 0.3, 0.2][:len(velocities_x)]
            weight_sum = sum(weights)
            
            velocity_x = sum(v * w for v, w in zip(velocities_x, weights)) / weight_sum
            velocity_y = sum(v * w for v, w in zip(velocities_y, weights)) / weight_sum
            
            # FPS kompanzasyonu - düşük FPS için hızı artır
            fps_multiplier = 30.0 / max(self.fps, 10.0)
            velocity_x *= fps_multiplier
            velocity_y *= fps_multiplier
        else:
            # Normal FPS için basit hesaplama
            p1 = history[-2]
            p2 = history[-1]
            velocity_x = p2.x - p1.x
            velocity_y = p2.y - p1.y

        # Tahminler
        predictions = []
        current_pos = history[-1]

        for step in range(1, steps_ahead + 1):
            # Doğrusal ekstrapolasyon
            next_x = current_pos.x + velocity_x
            next_y = current_pos.y + velocity_y

            next_pos = Point2D(next_x, next_y)
            predictions.append(next_pos)
            current_pos = next_pos

        return predictions

    def update_model(self, actual_position: Point2D, predicted_position: Point2D):
        """
        Tahmin modelini günceller.

        Args:
            actual_position: Gerçek pozisyon
            predicted_position: Tahmin edilen pozisyon
        """
        # Doğrusal model için güncelleme yok
        # Sadece geçmişi güncelle
        self._update_history(actual_position)


class KalmanMotionPredictor(BaseMotionPredictor, IMotionPredictor):
    """
    Kalman filtresi tabanlı hareket tahmini.

    Gürültülü ölçümlerle başa çıkabilir.
    Optimal tahmin sağlar.
    """

    def __init__(
        self, history_size: int = 10, process_noise: float = 0.01, 
        measurement_noise: float = 1.0, fps: float = 30.0
    ):
        """
        KalmanMotionPredictor constructor.

        Args:
            history_size: Geçmiş nokta sayısı
            process_noise: İşlem gürültüsü
            measurement_noise: Ölçüm gürültüsü
            fps: Video frame rate
        """
        super().__init__(history_size, fps)

        # FPS'e göre gürültü parametrelerini ayarla
        if fps < 20:
            # Düşük FPS için daha yüksek gürültü toleransı
            process_noise *= 2.0
            measurement_noise *= 1.5
        
        # Kalman filtresi parametreleri
        self.dt = self.frame_interval  # FPS'e göre zaman adımı

        # Durum: [x, y, vx, vy]
        self.state = np.zeros(4)

        # Durum geçiş matrisi
        self.F = np.array([[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Ölçüm matrisi
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # İşlem gürültü kovaryansı
        self.Q = process_noise * np.eye(4)

        # Ölçüm gürültü kovaryansı
        self.R = measurement_noise * np.eye(2)

        # Hata kovaryansı
        self.P = np.eye(4)

        self.initialized = False

    def predict(self, history: List[Point2D], steps_ahead: int = 1) -> List[Point2D]:
        """
        Gelecek pozisyonları tahmin eder.

        Args:
            history: Geçmiş pozisyonlar
            steps_ahead: Kaç adım ileri tahmin

        Returns:
            List[Point2D]: Tahmin edilen pozisyonlar
        """
        if not history:
            return [Point2D(0, 0)] * steps_ahead

        # İlk kullanımda başlat
        if not self.initialized and len(history) >= 2:
            # İlk durum tahmini
            self.state[0] = history[-1].x
            self.state[1] = history[-1].y
            self.state[2] = history[-1].x - history[-2].x  # vx
            self.state[3] = history[-1].y - history[-2].y  # vy
            self.initialized = True
        elif not self.initialized:
            # Tek nokta var, hız sıfır
            self.state[0] = history[-1].x
            self.state[1] = history[-1].y
            self.state[2] = 0
            self.state[3] = 0
            self.initialized = True

        # Mevcut ölçümle güncelle
        if history:
            measurement = np.array([history[-1].x, history[-1].y])
            self._kalman_update(measurement)

        # Tahminler
        predictions = []
        state_pred = self.state.copy()
        P_pred = self.P.copy()

        for step in range(steps_ahead):
            # Tahmin adımı
            state_pred = self.F @ state_pred
            P_pred = self.F @ P_pred @ self.F.T + self.Q

            # Tahmin edilen pozisyon
            pred_pos = Point2D(state_pred[0], state_pred[1])
            predictions.append(pred_pos)

        return predictions

    def update_model(self, actual_position: Point2D, predicted_position: Point2D):
        """
        Tahmin modelini günceller.

        Args:
            actual_position: Gerçek pozisyon
            predicted_position: Tahmin edilen pozisyon
        """
        # Kalman güncelleme
        measurement = np.array([actual_position.x, actual_position.y])
        self._kalman_update(measurement)

        # Geçmişi güncelle
        self._update_history(actual_position)

    def _kalman_update(self, measurement: np.ndarray):
        """
        Kalman filtresi güncelleme adımı.

        Args:
            measurement: Ölçüm vektörü [x, y]
        """
        # Tahmin adımı
        state_pred = self.F @ self.state
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Güncelleme adımı
        y = measurement - self.H @ state_pred  # İnovasyon
        S = self.H @ P_pred @ self.H.T + self.R  # İnovasyon kovaryansı
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # Kalman kazancı

        self.state = state_pred + K @ y
        self.P = (np.eye(4) - K @ self.H) @ P_pred


class SinusoidalMotionPredictor(BaseMotionPredictor, IMotionPredictor):
    """
    Sinüzoidal hareket tahmini.

    Kalp döngüsü gibi periyodik hareketler için.
    """

    def __init__(self, history_size: int = 30, expected_period: Optional[float] = None, fps: float = 30.0):
        """
        SinusoidalMotionPredictor constructor.

        Args:
            history_size: Geçmiş nokta sayısı
            expected_period: Beklenen periyot (frame)
            fps: Video frame rate
        """
        super().__init__(history_size, fps)
        # FPS'e göre periyot ayarla (kalp döngüsü ~1 saniye)
        self.expected_period = expected_period or int(fps)  # 1 saniye = fps frame
        self.amplitude_x = 0.0
        self.amplitude_y = 0.0
        self.phase = 0.0
        self.frequency = 2 * np.pi / self.expected_period
        self.center_x = 0.0
        self.center_y = 0.0

    def predict(self, history: List[Point2D], steps_ahead: int = 1) -> List[Point2D]:
        """
        Gelecek pozisyonları tahmin eder.

        Args:
            history: Geçmiş pozisyonlar
            steps_ahead: Kaç adım ileri tahmin

        Returns:
            List[Point2D]: Tahmin edilen pozisyonlar
        """
        if len(history) < self.expected_period // 2:
            # Yetersiz veri, doğrusal tahmin
            linear_predictor = LinearMotionPredictor()
            return linear_predictor.predict(history, steps_ahead)

        # Sinüzoidal parametreleri tahmin et
        self._estimate_sinusoidal_params(history)

        # Tahminler
        predictions = []
        current_time = len(history)

        for step in range(1, steps_ahead + 1):
            t = current_time + step

            # Sinüzoidal hareket
            x = self.center_x + self.amplitude_x * np.sin(self.frequency * t + self.phase)
            y = self.center_y + self.amplitude_y * np.sin(self.frequency * t + self.phase)

            predictions.append(Point2D(x, y))

        return predictions

    def update_model(self, actual_position: Point2D, predicted_position: Point2D):
        """
        Tahmin modelini günceller.

        Args:
            actual_position: Gerçek pozisyon
            predicted_position: Tahmin edilen pozisyon
        """
        # Geçmişi güncelle
        self._update_history(actual_position)

        # Yeterli veri varsa parametreleri yeniden tahmin et
        if len(self.position_history) >= self.expected_period:
            self._estimate_sinusoidal_params(self.position_history)

    def _estimate_sinusoidal_params(self, positions: List[Point2D]):
        """
        Sinüzoidal hareket parametrelerini tahmin et.

        Args:
            positions: Pozisyon geçmişi
        """
        # X ve Y koordinatlarını ayır
        x_coords = np.array([p.x for p in positions])
        y_coords = np.array([p.y for p in positions])

        # Merkez (ortalama)
        self.center_x = np.mean(x_coords)
        self.center_y = np.mean(y_coords)

        # Merkezden uzaklıklar
        x_centered = x_coords - self.center_x
        y_centered = y_coords - self.center_y

        # FFT ile frekans analizi
        if len(positions) >= 10:
            # X koordinatı için
            fft_x = np.fft.fft(x_centered)
            freqs = np.fft.fftfreq(len(x_centered))

            # En güçlü frekansı bul (DC hariç)
            power = np.abs(fft_x[1 : len(fft_x) // 2])
            if len(power) > 0:
                dominant_idx = np.argmax(power) + 1
                self.frequency = 2 * np.pi * freqs[dominant_idx]

                # Periyodu güncelle
                if self.frequency > 0:
                    self.expected_period = 2 * np.pi / self.frequency

        # Genlikler (RMS)
        self.amplitude_x = np.sqrt(np.mean(x_centered**2)) * np.sqrt(2)
        self.amplitude_y = np.sqrt(np.mean(y_centered**2)) * np.sqrt(2)

        # Faz tahmini (son nokta ile)
        if len(positions) > 0:
            last_x = positions[-1].x - self.center_x
            positions[-1].y - self.center_y

            if self.amplitude_x > 0:
                phase_x = np.arcsin(np.clip(last_x / self.amplitude_x, -1, 1))
            else:
                phase_x = 0

            t = len(positions) - 1
            self.phase = phase_x - self.frequency * t


class AdaptiveMotionPredictor(BaseMotionPredictor, IMotionPredictor):
    """
    Adaptif hareket tahmini.

    Hareket karakteristiğine göre otomatik olarak
    doğrusal veya sinüzoidal model seçer.
    """

    def __init__(self, history_size: int = 30, fps: float = 30.0):
        """
        AdaptiveMotionPredictor constructor.

        Args:
            history_size: Geçmiş nokta sayısı
            fps: Video frame rate
        """
        super().__init__(history_size, fps)
        self.linear_predictor = LinearMotionPredictor(history_size, fps)
        self.sinusoidal_predictor = SinusoidalMotionPredictor(history_size, None, fps)
        self.kalman_predictor = KalmanMotionPredictor(history_size, 0.01, 1.0, fps)

        self.motion_type = "linear"  # "linear", "sinusoidal", "complex"
        self.type_confidence = 0.0

    def predict(self, history: List[Point2D], steps_ahead: int = 1) -> List[Point2D]:
        """
        Gelecek pozisyonları tahmin eder.

        Args:
            history: Geçmiş pozisyonlar
            steps_ahead: Kaç adım ileri tahmin

        Returns:
            List[Point2D]: Tahmin edilen pozisyonlar
        """
        if len(history) < 5:
            # Az veri, doğrusal tahmin
            return self.linear_predictor.predict(history, steps_ahead)

        # Hareket tipini belirle
        self._detect_motion_type(history)

        # Uygun tahmin ediciyi kullan
        if self.motion_type == "sinusoidal":
            predictions = self.sinusoidal_predictor.predict(history, steps_ahead)
        elif self.motion_type == "complex":
            predictions = self.kalman_predictor.predict(history, steps_ahead)
        else:  # linear
            predictions = self.linear_predictor.predict(history, steps_ahead)

        # Düşük güven durumunda Kalman ile kombine et
        if self.type_confidence < 0.7:
            kalman_predictions = self.kalman_predictor.predict(history, steps_ahead)

            # Ağırlıklı ortalama
            combined = []
            for i in range(steps_ahead):
                weight = self.type_confidence
                x = predictions[i].x * weight + kalman_predictions[i].x * (1 - weight)
                y = predictions[i].y * weight + kalman_predictions[i].y * (1 - weight)
                combined.append(Point2D(x, y))

            return combined

        return predictions

    def update_model(self, actual_position: Point2D, predicted_position: Point2D):
        """
        Tahmin modelini günceller.

        Args:
            actual_position: Gerçek pozisyon
            predicted_position: Tahmin edilen pozisyon
        """
        # Tüm modelleri güncelle
        self.linear_predictor.update_model(actual_position, predicted_position)
        self.sinusoidal_predictor.update_model(actual_position, predicted_position)
        self.kalman_predictor.update_model(actual_position, predicted_position)

        # Geçmişi güncelle
        self._update_history(actual_position)

    def _detect_motion_type(self, positions: List[Point2D]):
        """
        Hareket tipini tespit et.

        Args:
            positions: Pozisyon geçmişi
        """
        if len(positions) < 10:
            self.motion_type = "linear"
            self.type_confidence = 1.0
            return

        # Hız değişimlerini hesapla
        velocities = []
        for i in range(1, len(positions)):
            vx = positions[i].x - positions[i - 1].x
            vy = positions[i].y - positions[i - 1].y
            velocities.append(np.sqrt(vx**2 + vy**2))

        velocities = np.array(velocities)

        # 1. Doğrusallık testi
        # Sabit hız = doğrusal hareket
        velocity_std = np.std(velocities)
        velocity_mean = np.mean(velocities)

        if velocity_mean > 0:
            cv = velocity_std / velocity_mean  # Varyasyon katsayısı
        else:
            cv = 0

        linearity_score = np.exp(-cv * 5)  # 0-1 arası

        # 2. Periyodiklik testi
        # Otokorelasyon
        if len(positions) >= 20:
            x_coords = np.array([p.x for p in positions])
            y_coords = np.array([p.y for p in positions])

            # Ortalamayı çıkar
            x_centered = x_coords - np.mean(x_coords)
            y_centered = y_coords - np.mean(y_coords)

            # Otokorelasyon
            x_corr = np.correlate(x_centered, x_centered, mode="full")
            y_corr = np.correlate(y_centered, y_centered, mode="full")

            # Normalize et
            x_corr = x_corr[len(x_corr) // 2 :] / x_corr[len(x_corr) // 2]
            y_corr = y_corr[len(y_corr) // 2 :] / y_corr[len(y_corr) // 2]

            # Periyodik peak'leri ara
            periodicity_score = 0.0

            if len(x_corr) > 10:
                # İlk peak'i bul (periyot)
                peaks = signal.find_peaks(x_corr[5:], height=0.5)[0]
                if len(peaks) > 0:
                    period = peaks[0] + 5
                    if 10 < period < 40:  # Makul periyot aralığı
                        periodicity_score = x_corr[period]
        else:
            periodicity_score = 0.0

        # Karar verme
        if linearity_score > 0.8:
            self.motion_type = "linear"
            self.type_confidence = linearity_score
        elif periodicity_score > 0.7:
            self.motion_type = "sinusoidal"
            self.type_confidence = periodicity_score
        else:
            self.motion_type = "complex"
            self.type_confidence = 0.5  # Orta güven


def create_motion_predictor(method: str, fps: float = 30.0, **kwargs) -> IMotionPredictor:
    """
    Hareket tahmini factory fonksiyonu.

    Args:
        method: Tahmin yöntemi
        fps: Video frame rate
        kwargs: Ek parametreler

    Returns:
        IMotionPredictor: Tahmin edici instance
    """
    # FPS'i kwargs'a ekle
    kwargs['fps'] = fps
    
    # FPS'e göre varsayılan parametreler
    if fps < 20:
        # Düşük FPS için optimize edilmiş parametreler
        if 'history_size' not in kwargs:
            kwargs['history_size'] = 15  # Daha fazla geçmiş
    
    if method == "linear":
        return LinearMotionPredictor(**kwargs)
    elif method == "kalman":
        return KalmanMotionPredictor(**kwargs)
    elif method == "sinusoidal":
        return SinusoidalMotionPredictor(**kwargs)
    elif method == "adaptive":
        return AdaptiveMotionPredictor(**kwargs)
    else:
        raise ValueError(f"Unknown motion prediction method: {method}")
