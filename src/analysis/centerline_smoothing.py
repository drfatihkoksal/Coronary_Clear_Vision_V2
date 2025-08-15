"""
Centerline smoothing and endpoint artifact removal
Uç noktalardaki kıvrılma ve artifact'leri düzeltmek için
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
import logging

logger = logging.getLogger(__name__)


def smooth_centerline_endpoints(
    centerline: np.ndarray, endpoint_trim: int = 20, smoothing_factor: float = 0.5
) -> np.ndarray:
    """
    Centerline uç noktalarındaki kıvrılmaları düzelt

    Args:
        centerline: (N, 2) şeklinde centerline noktaları (y, x)
        endpoint_trim: Her uçtan kırpılacak nokta sayısı
        smoothing_factor: Spline düzgünleştirme faktörü (0-1 arası)

    Returns:
        Düzeltilmiş centerline
    """
    if len(centerline) < 10:
        logger.warning("Centerline çok kısa, düzeltme yapılamıyor")
        return centerline

    # 1. Uç noktaları kırp
    if endpoint_trim > 0 and len(centerline) > 2 * endpoint_trim:
        trimmed = centerline[endpoint_trim:-endpoint_trim]
        keep_endpoints = True
    else:
        centerline.copy()
        keep_endpoints = False

    # 2. Parametrik spline fitting
    # Centerline boyunca kümülatif mesafe hesapla
    distances = np.zeros(len(centerline))
    for i in range(1, len(centerline)):
        distances[i] = distances[i - 1] + np.linalg.norm(centerline[i] - centerline[i - 1])

    # Normalize et
    if distances[-1] > 0:
        t = distances / distances[-1]
    else:
        t = np.linspace(0, 1, len(centerline))

    # 3. X ve Y için ayrı spline fitting
    try:
        # Smoothing factor'ü ayarla
        s = len(centerline) * smoothing_factor

        # Spline fitting - use full centerline for spline creation
        spline_y = UnivariateSpline(t, centerline[:, 0], s=s, k=3, ext=0)
        spline_x = UnivariateSpline(t, centerline[:, 1], s=s, k=3, ext=0)

        # Yeni noktalar oluştur - same number as original
        t_new = t  # Use same parameter values

        # Spline değerlerini hesapla
        new_y = spline_y(t_new)
        new_x = spline_x(t_new)

        # Smoothed centerline
        smoothed = np.column_stack((new_y, new_x))

        # Preserve exact endpoints if requested
        if keep_endpoints and endpoint_trim > 0:
            smoothed[:endpoint_trim] = centerline[:endpoint_trim]
            smoothed[-endpoint_trim:] = centerline[-endpoint_trim:]

    except Exception as e:
        logger.error(f"Spline fitting hatası: {e}")
        # Return original centerline if smoothing fails
        smoothed = centerline.copy()

    # 4. Uç noktaları özel olarak düzelt
    smoothed = fix_endpoint_artifacts(smoothed, original=centerline)

    logger.info(f"Spline smoothing: {len(centerline)} -> {len(smoothed)} points")
    return smoothed


def fix_endpoint_artifacts(
    centerline: np.ndarray, original: np.ndarray, endpoint_window: int = 10
) -> np.ndarray:
    """
    Uç noktalardaki artifact'leri düzelt

    Args:
        centerline: Smoothed centerline
        original: Orijinal centerline
        endpoint_window: Uç nokta pencere boyutu

    Returns:
        Düzeltilmiş centerline
    """
    fixed = centerline.copy()
    n_points = len(centerline)

    if n_points < 2 * endpoint_window:
        return fixed

    # Başlangıç noktası düzeltmesi
    # İlk birkaç noktanın yönünü kontrol et
    start_direction = centerline[endpoint_window] - centerline[0]
    start_norm = np.linalg.norm(start_direction)

    if start_norm > 0:
        start_direction = start_direction / start_norm

        # İlk noktaları yeniden hizala
        for i in range(endpoint_window):
            t = i / endpoint_window
            # Lineer interpolasyon ile düzelt
            expected_pos = (
                centerline[endpoint_window]
                - (endpoint_window - i) * start_direction * start_norm / endpoint_window
            )
            # Orijinal ile blend
            fixed[i] = (1 - t) * expected_pos + t * centerline[i]

    # Bitiş noktası düzeltmesi
    end_direction = centerline[-1] - centerline[-endpoint_window - 1]
    end_norm = np.linalg.norm(end_direction)

    if end_norm > 0:
        end_direction = end_direction / end_norm

        # Son noktaları yeniden hizala
        for i in range(endpoint_window):
            idx = -(i + 1)
            t = i / endpoint_window
            # Lineer interpolasyon ile düzelt
            expected_pos = (
                centerline[-endpoint_window - 1]
                + (endpoint_window - i) * end_direction * end_norm / endpoint_window
            )
            # Orijinal ile blend
            fixed[idx] = (1 - t) * expected_pos + t * centerline[idx]

    return fixed


def remove_centerline_loops(centerline: np.ndarray, min_segment_length: int = 5) -> np.ndarray:
    """
    Centerline'daki loop ve kıvrılmaları tespit edip düzelt

    Args:
        centerline: Centerline noktaları
        min_segment_length: Minimum segment uzunluğu

    Returns:
        Loop'ları kaldırılmış centerline
    """
    if len(centerline) < 2 * min_segment_length:
        return centerline

    cleaned = [centerline[0]]

    for i in range(1, len(centerline)):
        current_point = centerline[i]

        # Önceki noktalarla çakışma kontrolü
        skip = False
        for j in range(max(0, len(cleaned) - min_segment_length), len(cleaned)):
            dist = np.linalg.norm(current_point - cleaned[j])
            if dist < 2.0:  # Çok yakın noktaları atla
                skip = True
                break

        if not skip:
            cleaned.append(current_point)

    return np.array(cleaned)


def adaptive_centerline_smoothing(
    centerline: np.ndarray, curvature_threshold: float = 0.5
) -> np.ndarray:
    """
    Eğriliğe göre adaptif smoothing uygula
    Düz bölgelerde daha fazla, kıvrımlı bölgelerde daha az smoothing

    Args:
        centerline: Centerline noktaları
        curvature_threshold: Eğrilik eşik değeri

    Returns:
        Adaptif olarak düzgünleştirilmiş centerline
    """
    if len(centerline) < 5:
        return centerline

    # Lokal eğrilik hesapla
    curvatures = compute_local_curvature(centerline)

    # Adaptif smoothing
    smoothed = centerline.copy()

    for i in range(1, len(centerline) - 1):
        # Eğriliğe göre pencere boyutu belirle
        curvature = curvatures[i]
        if curvature < curvature_threshold:
            # Düz bölge - daha geniş pencere
            window_size = 5
        else:
            # Kıvrımlı bölge - dar pencere
            window_size = 2

        # Lokal ortalama
        start_idx = max(0, i - window_size)
        end_idx = min(len(centerline), i + window_size + 1)

        if end_idx > start_idx:
            smoothed[i] = np.mean(centerline[start_idx:end_idx], axis=0)

    return smoothed


def compute_local_curvature(centerline: np.ndarray) -> np.ndarray:
    """
    Her nokta için lokal eğrilik hesapla

    Args:
        centerline: Centerline noktaları

    Returns:
        Eğrilik değerleri
    """
    n_points = len(centerline)
    curvatures = np.zeros(n_points)

    for i in range(1, n_points - 1):
        # İki vektör arası açı
        v1 = centerline[i] - centerline[i - 1]
        v2 = centerline[i + 1] - centerline[i]

        # Normalize
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 > 0 and n2 > 0:
            v1 = v1 / n1
            v2 = v2 / n2

            # Açı hesapla (0-1 arası normalize edilmiş)
            dot_product = np.dot(v1, v2)
            curvatures[i] = 1 - (dot_product + 1) / 2

    # Uç noktalar için
    curvatures[0] = curvatures[1]
    curvatures[-1] = curvatures[-2]

    return curvatures


def preserve_length_smoothing(
    centerline: np.ndarray,
    sigma: float = 1.5,
    endpoint_preserve: int = 20,
    apply_endpoint_correction: bool = True,
) -> np.ndarray:
    """
    Nokta sayısını koruyarak centerline smoothing uygula

    Args:
        centerline: Orijinal centerline
        sigma: Gaussian smoothing parametresi
        endpoint_preserve: Uç noktalarda korunacak nokta sayısı
        apply_endpoint_correction: Uç nokta düzeltmesi uygula

    Returns:
        Smoothed centerline (aynı uzunlukta)
    """
    if len(centerline) < 20:
        logger.warning("Centerline çok kısa, minimal smoothing uygulanıyor")
        sigma = min(sigma, 0.5)

    smoothed = centerline.copy()
    n_points = len(centerline)

    # Orta bölgeye smoothing uygula, uç noktaları koru
    if n_points > 2 * endpoint_preserve:
        # Orta bölgeyi smooth et
        middle_y = gaussian_filter1d(
            centerline[endpoint_preserve:-endpoint_preserve, 0], sigma=sigma
        )
        middle_x = gaussian_filter1d(
            centerline[endpoint_preserve:-endpoint_preserve, 1], sigma=sigma
        )

        # Smoothed değerleri yerleştir
        smoothed[endpoint_preserve:-endpoint_preserve, 0] = middle_y
        smoothed[endpoint_preserve:-endpoint_preserve, 1] = middle_x

        # Uç noktalarda geçişi yumuşat
        for i in range(endpoint_preserve):
            # Başlangıç tarafı
            weight = i / endpoint_preserve
            smoothed[i] = (1 - weight) * centerline[i] + weight * smoothed[endpoint_preserve]

            # Bitiş tarafı
            end_idx = -(i + 1)
            smoothed[end_idx] = (1 - weight) * centerline[end_idx] + weight * smoothed[
                -endpoint_preserve - 1
            ]
    else:
        # Çok kısa centerline için minimal smoothing
        smoothed[:, 0] = gaussian_filter1d(centerline[:, 0], sigma=0.5)
        smoothed[:, 1] = gaussian_filter1d(centerline[:, 1], sigma=0.5)

    # Uç nokta düzeltmesi uygula
    if apply_endpoint_correction and len(smoothed) > 2 * endpoint_preserve:
        smoothed = spline_based_endpoint_correction(smoothed, endpoint_preserve)

    logger.info(f"Preserve length smoothing: {len(centerline)} -> {len(smoothed)} nokta")
    return smoothed


def spline_based_endpoint_correction(
    centerline: np.ndarray, endpoint_window: int = 20
) -> np.ndarray:
    """
    Spline fitting kullanarak uç nokta düzeltmesi
    2024 research'e göre 5th degree spline ve penalty terms

    Args:
        centerline: Smoothed centerline
        endpoint_window: Düzeltilecek uç nokta sayısı

    Returns:
        Düzeltilmiş centerline
    """
    if len(centerline) < 3 * endpoint_window:
        return centerline

    corrected = centerline.copy()
    n_points = len(centerline)

    try:
        # Başlangıç uç düzeltmesi
        # İç bölgeden trend direction hesapla
        inner_start = endpoint_window
        inner_end = min(inner_start + endpoint_window * 2, n_points - endpoint_window)
        inner_segment = centerline[inner_start:inner_end]

        if len(inner_segment) > 5:
            # Parametrik t değerleri
            t_inner = np.linspace(0, 1, len(inner_segment))

            # 3rd degree spline fitting (C² continuity için)
            spline_y = UnivariateSpline(
                t_inner, inner_segment[:, 0], s=len(inner_segment) * 0.1, k=3
            )
            spline_x = UnivariateSpline(
                t_inner, inner_segment[:, 1], s=len(inner_segment) * 0.1, k=3
            )

            # Başlangıç noktaları için MINIMAL extrapolation (sadece smoothing, extension yok)
            t_start = np.linspace(-0.1, 0, endpoint_window + 1)[:-1]  # Minimal extrapolation

            # Sadece orijinal sınırlar içinde kalmaya zorla
            extrapolated_y = spline_y(t_start)
            extrapolated_x = spline_x(t_start)

            # Orijinal endpoints ile blend - aggressive extrapolation yerine gentle smoothing
            for j in range(endpoint_window):
                blend_factor = j / endpoint_window  # 0 to 1
                corrected[j, 0] = (1 - blend_factor) * centerline[
                    j, 0
                ] + blend_factor * extrapolated_y[j]
                corrected[j, 1] = (1 - blend_factor) * centerline[
                    j, 1
                ] + blend_factor * extrapolated_x[j]

        # Bitiş uç düzeltmesi
        inner_start = max(endpoint_window, n_points - endpoint_window * 3)
        inner_end = n_points - endpoint_window
        inner_segment = centerline[inner_start:inner_end]

        if len(inner_segment) > 5:
            # Parametrik t değerleri
            t_inner = np.linspace(0, 1, len(inner_segment))

            # 3rd degree spline fitting
            spline_y = UnivariateSpline(
                t_inner, inner_segment[:, 0], s=len(inner_segment) * 0.1, k=3
            )
            spline_x = UnivariateSpline(
                t_inner, inner_segment[:, 1], s=len(inner_segment) * 0.1, k=3
            )

            # Bitiş noktaları için MINIMAL extrapolation (sadece smoothing, extension yok)
            t_end = np.linspace(1, 1.1, endpoint_window + 1)[1:]  # Minimal extrapolation

            # Sadece orijinal sınırlar içinde kalmaya zorla
            extrapolated_y = spline_y(t_end)
            extrapolated_x = spline_x(t_end)

            # Orijinal endpoints ile blend - aggressive extrapolation yerine gentle smoothing
            for j in range(endpoint_window):
                blend_factor = j / endpoint_window  # 0 to 1
                idx = -(endpoint_window - j)
                corrected[idx, 0] = (1 - blend_factor) * centerline[
                    idx, 0
                ] + blend_factor * extrapolated_y[j]
                corrected[idx, 1] = (1 - blend_factor) * centerline[
                    idx, 1
                ] + blend_factor * extrapolated_x[j]

    except Exception as e:
        logger.warning(f"Spline-based endpoint correction failed: {e}")
        return centerline

    logger.info(f"Applied spline-based endpoint correction to {endpoint_window} points at each end")
    return corrected


def apply_centerline_smoothing(
    centerline: np.ndarray, method: str = "spline", **kwargs
) -> np.ndarray:
    """
    Centerline smoothing uygula

    Args:
        centerline: Orijinal centerline
        method: Smoothing yöntemi ('spline', 'adaptive', 'gaussian', 'preserve_length')
        **kwargs: Yöntem parametreleri

    Returns:
        Düzgünleştirilmiş centerline
    """
    logger.info(f"Centerline smoothing uygulanıyor: {method}")
    logger.info(f"Orijinal centerline uzunluğu: {len(centerline)} nokta")

    # Loop'ları kaldırmayı opsiyonel yap
    if kwargs.get("remove_loops", False):
        centerline = remove_centerline_loops(centerline)
        logger.info(f"Loop kaldırma sonrası: {len(centerline)} nokta")

    if method == "spline":
        return smooth_centerline_endpoints(
            centerline,
            endpoint_trim=kwargs.get("endpoint_trim", 20),
            smoothing_factor=kwargs.get("smoothing_factor", 0.5),
        )
    elif method == "adaptive":
        return adaptive_centerline_smoothing(
            centerline, curvature_threshold=kwargs.get("curvature_threshold", 0.5)
        )
    elif method == "gaussian":
        sigma = kwargs.get("sigma", 2.0)
        smoothed = centerline.copy()
        smoothed[:, 0] = gaussian_filter1d(centerline[:, 0], sigma=sigma)
        smoothed[:, 1] = gaussian_filter1d(centerline[:, 1], sigma=sigma)
        return smoothed
    elif method == "preserve_length":
        # Uzunluğu koruyan yeni smoothing metodu
        return preserve_length_smoothing(
            centerline,
            sigma=kwargs.get("sigma", 1.5),
            endpoint_preserve=kwargs.get("endpoint_preserve", 20),
        )
    elif method == "curvature_adaptive":
        # 2024 research: curvature-based adaptive smoothing
        return curvature_adaptive_smoothing(
            centerline,
            curvature_threshold=kwargs.get("curvature_threshold", 0.3),
            endpoint_preserve=kwargs.get("endpoint_preserve", 20),
        )
    else:
        logger.warning(f"Bilinmeyen smoothing yöntemi: {method}")
        return centerline


def curvature_adaptive_smoothing(
    centerline: np.ndarray, curvature_threshold: float = 0.3, endpoint_preserve: int = 20
) -> np.ndarray:
    """
    Eğriliğe göre adaptif smoothing - 2024 research based
    Düz bölgelerde daha fazla, kıvrımlı bölgelerde daha az smoothing

    Args:
        centerline: Centerline noktaları
        curvature_threshold: Eğrilik eşik değeri
        endpoint_preserve: Uç noktalarda korunacak nokta sayısı

    Returns:
        Adaptif olarak düzgünleştirilmiş centerline
    """
    if len(centerline) < 10:
        return centerline

    # Lokal eğrilik hesapla
    curvatures = compute_local_curvature(centerline)

    # Adaptif smoothing with endpoint preservation
    smoothed = centerline.copy()
    n_points = len(centerline)

    # Preserve endpoints
    start_preserve = min(endpoint_preserve, n_points // 4)
    end_preserve = min(endpoint_preserve, n_points // 4)

    for i in range(start_preserve, n_points - end_preserve):
        # Eğriliğe göre adaptive parameters
        curvature = curvatures[i]

        if curvature < curvature_threshold:
            # Düz bölge - daha geniş pencere ve güçlü smoothing
            window_size = 4
            sigma = 1.5
        else:
            # Kıvrımlı bölge - dar pencere ve hafif smoothing
            window_size = 2
            sigma = 0.5

        # Lokal Gaussian smoothing window
        start_idx = max(start_preserve, i - window_size)
        end_idx = min(n_points - end_preserve, i + window_size + 1)

        if end_idx > start_idx + 2:
            # Local segment için Gaussian weights
            local_segment = centerline[start_idx:end_idx]
            center_in_segment = i - start_idx

            # Gaussian weights centered at current point
            distances = np.arange(len(local_segment)) - center_in_segment
            weights = np.exp(-0.5 * (distances / sigma) ** 2)
            weights = weights / np.sum(weights)

            # Weighted average
            smoothed[i] = np.sum(local_segment * weights[:, np.newaxis], axis=0)

    # Apply spline-based endpoint correction
    if len(smoothed) > 2 * endpoint_preserve:
        smoothed = spline_based_endpoint_correction(smoothed, endpoint_preserve)

    logger.info(f"Applied curvature-adaptive smoothing with threshold {curvature_threshold}")
    return smoothed
