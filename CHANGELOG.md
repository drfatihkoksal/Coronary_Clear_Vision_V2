# Changelog

All notable changes to Coronary Clear Vision V2 will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-01-13

### 🎉 Initial Release

#### ✨ Features
- **AngioPy AI Segmentation**: Deep learning-based vessel segmentation using InceptionResNetV2 model
- **QCA Analysis**: Quantitative Coronary Analysis with sub-pixel accuracy
- **RWS Calculation**: Radial Wall Strain analysis across cardiac cycles
- **Multi-frame DICOM Support**: Full video analysis capability
- **Frame-to-Frame Tracking**: Temporal coherence through template matching
- **Cardiac Phase Detection**: ECG-synchronized analysis (D1, D2, S1, S2 phases)
- **Hampel Filter**: Advanced outlier detection for frame quality assessment
- **Auto-calibration**: Catheter-based pixel-to-mm conversion
- **Enhanced Visualization**: Real-time overlay of segmentation and measurements

#### 🛠️ Technical Stack
- Python 3.9+ with PyQt6 GUI framework
- PyTorch for deep learning inference
- OpenCV for image processing
- Matplotlib for data visualization
- SQLite for data persistence

#### 📊 Analysis Capabilities
- Vessel diameter measurements
- Stenosis percentage calculation
- Minimum Lumen Diameter (MLD) detection
- Reference vessel diameter computation
- Lesion length measurement
- Frame-by-frame diameter profiles
- Statistical analysis (mean, std dev, CV)

#### 🔧 Configuration
- Flexible settings via JSON configuration
- Support for 5F, 6F, 7F, 8F catheter sizes
- Customizable analysis parameters

#### 📝 Documentation
- Comprehensive README with installation guide
- Academic citation support (CITATION.cff)
- Dual licensing (Academic + Commercial)
- Demo video included

### ⚠️ Known Limitations
- Research use only (not FDA/CE approved)
- Requires CUDA-capable GPU for optimal performance
- Model weights auto-downloaded on first use (~200MB)

### 🙏 Acknowledgments
- AngioPy team for the segmentation model
- Open-source community for dependencies

---

For questions or issues, please visit: https://github.com/drfatihkoksal/Coronary_Clear_Vision_V2/issues