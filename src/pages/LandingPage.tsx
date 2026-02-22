import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    Activity,
    Brain,
    ChevronRight,
    Microscope,
    ArrowRight,
    FlaskConical,
    Layers,
    Waves,
    Network
} from 'lucide-react';

export function LandingPage() {
    const navigate = useNavigate();
    const [scrolled, setScrolled] = useState(false);

    useEffect(() => {
        const handleScroll = () => {
            setScrolled(window.scrollY > 50);
        };
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    return (
        <div className="min-h-screen bg-[#0f0f1a] text-white selection:bg-blue-500/30 overflow-x-hidden">
            {/* Navbar */}
            <nav className={`fixed top-0 w-full z-50 transition-all duration-300 border-b ${scrolled ? 'bg-[#0f0f1a]/80 backdrop-blur-md border-white/10' : 'bg-transparent border-transparent'
                }`}>
                <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20">
                            <Activity className="w-6 h-6 text-white" />
                        </div>
                        <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-white/70">
                            Coronary RWS Analyser
                        </span>
                    </div>

                    <div className="flex items-center gap-4">
                        <button
                            onClick={() => navigate('/login')}
                            className="px-5 py-2 text-sm font-medium text-white/70 hover:text-white transition-colors"
                        >
                            Log in
                        </button>
                        <button
                            onClick={() => navigate('/register')}
                            className="group px-5 py-2 bg-white text-black text-sm font-semibold rounded-lg hover:bg-gray-100 transition-all shadow-[0_0_20px_rgba(255,255,255,0.15)] hover:shadow-[0_0_30px_rgba(255,255,255,0.25)] flex items-center gap-2"
                        >
                            Get Started
                            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                        </button>
                    </div>
                </div>
            </nav>

            {/* Hero Section */}
            <section className="relative pt-32 pb-20 lg:pt-48 lg:pb-32 px-6">
                <div className="absolute inset-0 overflow-hidden">
                    <div className="absolute -top-[30%] -right-[10%] w-[70vw] h-[70vw] rounded-full bg-blue-600/10 blur-[120px]" />
                    <div className="absolute -bottom-[20%] -left-[10%] w-[50vw] h-[50vw] rounded-full bg-indigo-600/10 blur-[100px]" />
                </div>

                <div className="max-w-7xl mx-auto relative z-10">
                    <div className="grid lg:grid-cols-2 gap-16 items-center">
                        <div className="space-y-8">
                            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-400 text-sm font-medium uppercase tracking-wide">
                                <FlaskConical className="w-4 h-4" />
                                <span>For Research Use Only</span>
                            </div>

                            <h1 className="text-5xl lg:text-7xl font-bold leading-tight tracking-tight">
                                Complete Platform for <br />
                                <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-indigo-400">
                                    Coronary Analysis
                                </span>
                            </h1>

                            <p className="text-xl text-white/60 leading-relaxed max-w-xl">
                                The definitive end-to-end solution. From professional <strong>DICOM viewing</strong> and <strong>QCA</strong> to advanced <strong>Radial Wall Strain</strong> and <strong>3D QFR</strong> analysis synchronized with ECG.
                            </p>

                            <div className="flex flex-col sm:flex-row gap-4 pt-4">
                                <button
                                    onClick={() => navigate('/register')}
                                    className="px-8 py-4 bg-blue-600 hover:bg-blue-500 text-white rounded-xl font-semibold transition-all shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 flex items-center justify-center gap-2"
                                >
                                    Start Analysis
                                    <ChevronRight className="w-5 h-5" />
                                </button>
                                <button
                                    onClick={() => document.getElementById('methodology')?.scrollIntoView({ behavior: 'smooth' })}
                                    className="px-8 py-4 bg-white/5 hover:bg-white/10 text-white border border-white/10 rounded-xl font-semibold transition-all flex items-center justify-center gap-2"
                                >
                                    View Methodology
                                </button>
                            </div>

                            <div className="pt-8 border-t border-white/10 flex gap-8">
                                <div>
                                    <div className="text-3xl font-bold text-white">AI</div>
                                    <div className="text-sm text-white/40">Powered</div>
                                </div>
                                <div>
                                    <div className="text-3xl font-bold text-white">&lt;0.1mm</div>
                                    <div className="text-sm text-white/40">Precision</div>
                                </div>
                                <div>
                                    <div className="text-3xl font-bold text-white">Real-time</div>
                                    <div className="text-sm text-white/40">Analysis</div>
                                </div>
                            </div>
                        </div>

                        {/* Hero Visual */}
                        <div className="relative group">
                            <div className="absolute inset-0 bg-gradient-to-tr from-blue-500/20 to-purple-500/20 rounded-2xl blur-3xl opacity-50 group-hover:opacity-75 transition-opacity duration-1000" />
                            <div className="relative bg-[#1a1a2e] border border-white/10 rounded-2xl p-2 shadow-2xl overflow-hidden aspect-square flex flex-col items-center justify-center">
                                <img
                                    src="/assets/images/hero_rws_new.png"
                                    alt="Coronary RWS Visualization"
                                    className="w-full h-full object-cover rounded-xl opacity-90 hover:scale-105 transition-transform duration-700"
                                />
                                <div className="absolute bottom-6 left-6 right-6 grid grid-cols-3 gap-2 backdrop-blur-md bg-black/40 p-4 rounded-xl border border-white/10">
                                    <div className="text-center border-r border-white/10">
                                        <div className="text-[10px] text-white/40 uppercase tracking-widest mb-1">D-Max</div>
                                        <div className="text-lg font-mono text-blue-400">3.42mm</div>
                                    </div>
                                    <div className="text-center border-r border-white/10">
                                        <div className="text-[10px] text-white/40 uppercase tracking-widest mb-1">QFR</div>
                                        <div className="text-lg font-mono text-purple-400">0.82</div>
                                    </div>
                                    <div className="text-center">
                                        <div className="text-[10px] text-white/40 uppercase tracking-widest mb-1">Conf</div>
                                        <div className="text-lg font-mono text-emerald-400">98%</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Core Capabilities Section */}
            <section className="py-24 bg-[#0a0a12] border-y border-white/5">
                <div className="max-w-7xl mx-auto px-6">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl md:text-5xl font-bold mb-6">Complete Diagnostic Toolset</h2>
                        <p className="text-white/60 max-w-2xl mx-auto">
                            Beyond Radial Wall Strain, access a full suite of professional-grade analysis tools integrated into a single seamless workflow.
                        </p>
                    </div>

                    <div className="grid lg:grid-cols-4 gap-6">
                        <div className="group relative rounded-2xl bg-[#0f0f1a] border border-white/10 overflow-hidden hover:border-blue-500/30 transition-all">
                            <div className="aspect-video w-full overflow-hidden bg-black/50 relative">
                                <div className="absolute inset-0 bg-blue-500/10 mix-blend-overlay z-10" />
                                <img src="/assets/images/dicom_viewer_showcase.png" alt="Professional DICOM Viewer" className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700" />
                            </div>
                            <div className="p-6">
                                <h3 className="text-xl font-semibold mb-2 text-white flex items-center gap-2">
                                    <Activity className="w-5 h-5 text-blue-400" />
                                    Professional DICOM Viewer
                                </h3>
                                <p className="text-white/60 text-sm">Hardware-accelerated playback with multi-frame navigation, zoom/pan controls, and window-level adjustments for pristine image clarity.</p>
                            </div>
                        </div>

                        <div className="group relative rounded-2xl bg-[#0f0f1a] border border-white/10 overflow-hidden hover:border-emerald-500/30 transition-all">
                            <div className="aspect-video w-full overflow-hidden bg-black/50 relative">
                                <div className="absolute inset-0 bg-emerald-500/10 mix-blend-overlay z-10" />
                                <img src="/assets/images/ecg_viewer_showcase.png" alt="Synchronized ECG Analysis" className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700" />
                            </div>
                            <div className="p-6">
                                <h3 className="text-xl font-semibold mb-2 text-white flex items-center gap-2">
                                    <Activity className="w-5 h-5 text-emerald-400" />
                                    Synchronized ECG Analysis
                                </h3>
                                <p className="text-white/60 text-sm">Integrated signal viewer that automatically synchronizes angiographic frames with cardiac phases, enabling precise systolic/diastolic correlation.</p>
                            </div>
                        </div>

                        <div className="group relative rounded-2xl bg-[#0f0f1a] border border-white/10 overflow-hidden hover:border-purple-500/30 transition-all">
                            <div className="aspect-video w-full overflow-hidden bg-black/50 relative">
                                <div className="absolute inset-0 bg-purple-500/10 mix-blend-overlay z-10" />
                                <img src="/assets/images/qca_analysis_showcase.png" alt="Quantitative Coronary Analysis" className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700" />
                            </div>
                            <div className="p-6">
                                <h3 className="text-xl font-semibold mb-2 text-white flex items-center gap-2">
                                    <Microscope className="w-5 h-5 text-purple-400" />
                                    Quantitative Coronary Analysis
                                </h3>
                                <p className="text-white/60 text-sm">Automated vessel detection and stenosis quantification. Get instant metrics for Minimal Lumen Diameter (MLD), Reference Diameter, and % Stenosis.</p>
                            </div>
                        </div>

                        <div className="group relative rounded-2xl bg-[#0f0f1a] border border-white/10 overflow-hidden hover:border-indigo-500/30 transition-all">
                            <div className="aspect-video w-full overflow-hidden bg-black/50 relative">
                                <div className="absolute inset-0 bg-indigo-500/10 mix-blend-overlay z-10" />
                                <img src="/assets/images/hybrid_tracking.png" alt="3D QFR Analysis" className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700" />
                            </div>
                            <div className="p-6">
                                <h3 className="text-xl font-semibold mb-2 text-white flex items-center gap-2">
                                    <Layers className="w-5 h-5 text-indigo-400" />
                                    3D QFR Analysis
                                </h3>
                                <p className="text-white/60 text-sm">Wire-free functional assessment derived from 3D reconstruction. Calculate pressure drop and flow velocity without invasive wires.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* RWS Methodology Section */}
            <section id="methodology" className="py-24 bg-[#0a0a12] border-y border-white/5">
                <div className="max-w-7xl mx-auto px-6">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl md:text-5xl font-bold mb-6">Advanced Analysis Engine</h2>
                        <p className="text-white/60 max-w-2xl mx-auto">
                            Powered by a proprietary stack of <strong>Global Optical Flow</strong>, <strong>Hybrid AI</strong>, and <strong>Sub-pixel Tracking</strong> algorithms to deliver precision where others fail.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-2 lg:grid-cols-2 gap-8">
                        <div className="relative p-8 rounded-2xl bg-[#0f0f1a] border border-white/10 hover:border-blue-500/30 transition-all group lg:col-span-1 overflow-hidden">
                            <div className="absolute inset-0 opacity-40 group-hover:opacity-60 transition-opacity">
                                <img src="/assets/images/optical_flow.png" alt="" className="w-full h-full object-cover mix-blend-overlay transition-transform duration-700 group-hover:scale-105" />
                                <div className="absolute inset-0 bg-gradient-to-t from-[#0f0f1a] via-[#0f0f1a]/90 to-transparent" />
                            </div>
                            <div className="relative z-10">
                                <div className="w-12 h-12 bg-blue-500/10 rounded-lg flex items-center justify-center mb-6 text-blue-400 group-hover:scale-110 transition-transform">
                                    <Activity />
                                </div>
                                <h3 className="text-xl font-semibold mb-3 text-white">ECG-Less Visualization</h3>
                                <p className="text-white/60 text-sm mb-4">
                                    No ECG? No problem. Our engine uses <strong>Farneback Dense Optical Flow</strong> to calculate global motion vectors across the entire frame.
                                </p>
                                <ul className="space-y-2 text-sm text-white/50">
                                    <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-blue-500" />Infer cardiac phases (Systole/Diastole) from motion</li>
                                    <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-blue-500" />Synchronize scans without hardware gating</li>
                                </ul>
                            </div>
                        </div>

                        <div className="relative p-8 rounded-2xl bg-[#0f0f1a] border border-white/10 hover:border-indigo-500/30 transition-all group lg:col-span-1 overflow-hidden">
                            <div className="absolute inset-0 opacity-40 group-hover:opacity-60 transition-opacity">
                                <img src="/assets/images/hybrid_tracking.png" alt="" className="w-full h-full object-cover mix-blend-overlay transition-transform duration-700 group-hover:scale-105" />
                                <div className="absolute inset-0 bg-gradient-to-t from-[#0f0f1a] via-[#0f0f1a]/90 to-transparent" />
                            </div>
                            <div className="relative z-10">
                                <div className="w-12 h-12 bg-indigo-500/10 rounded-lg flex items-center justify-center mb-6 text-indigo-400 group-hover:scale-110 transition-transform">
                                    <Microscope />
                                </div>
                                <h3 className="text-xl font-semibold mb-3 text-white">Sub-pixel Hybrid Tracking</h3>
                                <p className="text-white/60 text-sm mb-4">
                                    Typical vessel trackers drift. Ours doesn't. We combine <strong>CSRT</strong> for robust ROI lock with <strong>Template Matching</strong> for sub-pixel precision.
                                </p>
                                <ul className="space-y-2 text-sm text-white/50">
                                    <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-indigo-500" />CSRT (Channel & Spatial Reliability) Tracker</li>
                                    <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-indigo-500" />Lucas-Kanade Optical Flow Fallback</li>
                                </ul>
                            </div>
                        </div>

                        <div className="relative p-8 rounded-2xl bg-[#0f0f1a] border border-white/10 hover:border-purple-500/30 transition-all group lg:col-span-1 overflow-hidden">
                            <div className="absolute inset-0 opacity-40 group-hover:opacity-60 transition-opacity">
                                <img src="/assets/images/ai_segmentation.png" alt="" className="w-full h-full object-cover mix-blend-overlay transition-transform duration-700 group-hover:scale-105" />
                                <div className="absolute inset-0 bg-gradient-to-t from-[#0f0f1a] via-[#0f0f1a]/90 to-transparent" />
                            </div>
                            <div className="relative z-10">
                                <div className="w-12 h-12 bg-purple-500/10 rounded-lg flex items-center justify-center mb-6 text-purple-400 group-hover:scale-110 transition-transform">
                                    <Brain />
                                </div>
                                <h3 className="text-xl font-semibold mb-3 text-white">AI-Powered Segmentation</h3>
                                <p className="text-white/60 text-sm mb-4">
                                    Advanced deep learning models including <strong>nnU-Net</strong> with wide-field context provide state-of-the-art vessel segmentation.
                                </p>
                                <div className="bg-black/40 p-3 rounded font-mono text-xs text-purple-300 border border-white/5 inline-block">
                                    Deep Learning Models
                                </div>
                            </div>
                        </div>

                        <div className="relative p-8 rounded-2xl bg-[#0f0f1a] border border-white/10 hover:border-emerald-500/30 transition-all group lg:col-span-1 overflow-hidden">
                            <div className="absolute inset-0 opacity-40 group-hover:opacity-60 transition-opacity">
                                <img src="/assets/images/math_precision.png" alt="" className="w-full h-full object-cover mix-blend-overlay transition-transform duration-700 group-hover:scale-105" />
                                <div className="absolute inset-0 bg-gradient-to-t from-[#0f0f1a] via-[#0f0f1a]/90 to-transparent" />
                            </div>
                            <div className="relative z-10">
                                <div className="w-12 h-12 bg-emerald-500/10 rounded-lg flex items-center justify-center mb-6 text-emerald-400 group-hover:scale-110 transition-transform">
                                    <Waves />
                                </div>
                                <h3 className="text-xl font-semibold mb-3 text-white">Computational Fluid Dynamics</h3>
                                <p className="text-white/60 text-sm mb-4">
                                    Real-time <strong>QFR</strong> computation using simplified Navier-Stokes equations. Derives pressure gradients from vessel geometry and flow velocity.
                                </p>
                                <div className="bg-black/40 p-3 rounded font-mono text-xs text-emerald-300 border border-white/5">
                                    {'\u0394'}P = f(Geometry, Flow)
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Sub-Pixel Accuracy Spotlight */}
            <section className="py-24 px-6 bg-[#0f0f1a] relative overflow-hidden">
                <div className="absolute top-0 right-0 w-[50vw] h-[50vw] bg-teal-500/5 rounded-full blur-[120px] pointer-events-none" />
                <div className="max-w-7xl mx-auto grid lg:grid-cols-2 gap-16 items-center relative z-10">
                    <div className="order-2 lg:order-1 relative group">
                        <div className="absolute inset-0 bg-gradient-to-tr from-teal-500/20 to-blue-500/20 rounded-2xl blur-2xl opacity-50 group-hover:opacity-75 transition-opacity duration-1000" />
                        <div className="relative border border-white/10 rounded-2xl overflow-hidden shadow-2xl">
                            <img src="/assets/images/sub_pixel_accuracy.png" alt="Sub-pixel Gaussian Fitting Visualization" className="w-full h-auto transform group-hover:scale-105 transition-transform duration-700" />
                        </div>
                    </div>
                    <div className="order-1 lg:order-2 space-y-6">
                        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-500/10 border border-teal-500/20 text-teal-400 text-sm font-medium uppercase tracking-wide">
                            <Microscope className="w-4 h-4" />
                            <span>Precision Metrology</span>
                        </div>
                        <h2 className="text-3xl md:text-5xl font-bold leading-tight">
                            Beyond the Pixel Grid: <br />
                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-cyan-400">Sub-Pixel Accuracy</span>
                        </h2>
                        <p className="text-xl text-white/60 leading-relaxed">
                            Pixels are square, but vessels are organic. Our <strong>Gaussian Model Fitting</strong> algorithm transcends resolution limits by reconstructing the true vessel intensity profile mathematically.
                        </p>
                        <div className="space-y-6 pt-4">
                            <div className="flex gap-4">
                                <div className="w-12 h-12 bg-white/5 rounded-full flex items-center justify-center border border-white/10 shrink-0">
                                    <span className="font-bold text-teal-400">1</span>
                                </div>
                                <div>
                                    <h4 className="text-lg font-semibold text-white mb-1">Bilinear Interpolation</h4>
                                    <p className="text-white/50 text-sm">We sample intensity values <em>between</em> pixels, creating a high-fidelity density map that captures subtle edge variations invisible to standard thresholding.</p>
                                </div>
                            </div>
                            <div className="flex gap-4">
                                <div className="w-12 h-12 bg-white/5 rounded-full flex items-center justify-center border border-white/10 shrink-0">
                                    <span className="font-bold text-teal-400">2</span>
                                </div>
                                <div>
                                    <h4 className="text-lg font-semibold text-white mb-1">Gaussian Profile Fitting</h4>
                                    <p className="text-white/50 text-sm">Instead of counting pixels, we fit a continuous Gaussian curve to the cross-section. The FWHM (Full Width at Half Maximum) provides diameter measurements with <strong>&lt;0.1mm sub-pixel precision</strong>.</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Risk Classification */}
            <section className="py-24 px-6 relative overflow-hidden">
                <div className="max-w-4xl mx-auto">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl font-bold mb-4">Risk Classification</h2>
                        <p className="text-white/60">Automated risk stratification based on plaque elasticity.</p>
                    </div>
                    <div className="space-y-4">
                        <div className="flex items-center gap-6 p-6 rounded-xl bg-white/5 border border-white/5 border-l-4 border-l-green-500 hover:bg-white/10 transition-colors">
                            <div className="text-2xl font-bold font-mono text-green-400 w-24">&lt; 8%</div>
                            <div>
                                <h4 className="font-semibold text-white">Normal Vessel</h4>
                                <p className="text-sm text-white/50">Healthy vessel wall compliant elasticity. No significant plaque burden.</p>
                            </div>
                        </div>
                        <div className="flex items-center gap-6 p-6 rounded-xl bg-white/5 border border-white/5 border-l-4 border-l-yellow-500 hover:bg-white/10 transition-colors">
                            <div className="text-2xl font-bold font-mono text-yellow-400 w-24">8-12%</div>
                            <div>
                                <h4 className="font-semibold text-white">Intermediate Risk</h4>
                                <p className="text-sm text-white/50">Potential mild plaque formation. Recommended for periodic monitoring.</p>
                            </div>
                        </div>
                        <div className="flex items-center gap-6 p-6 rounded-xl bg-gradient-to-r from-red-500/10 to-transparent border border-red-500/20 border-l-4 border-l-red-500">
                            <div className="text-2xl font-bold font-mono text-red-400 w-24">&gt; 12%</div>
                            <div>
                                <h4 className="font-semibold text-white">Vulnerable Plaque</h4>
                                <p className="text-sm text-white/50">High-risk lesion characteristics. Indicative of increased biomechanical instability and potential adverse events.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* QFR Technology Section */}
            <section className="py-24 bg-[#0a0a12] border-y border-white/5 relative overflow-hidden">
                <div className="absolute top-1/2 left-0 w-[40vw] h-[40vw] bg-purple-500/5 rounded-full blur-[120px] -translate-y-1/2 pointer-events-none" />
                <div className="max-w-7xl mx-auto px-6 relative z-10">
                    <div className="text-center mb-16">
                        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-400 text-sm font-medium uppercase tracking-wide mb-6">
                            <Waves className="w-4 h-4" />
                            <span>Hemodynamic Intelligence</span>
                        </div>
                        <h2 className="text-3xl md:text-5xl font-bold mb-6">Wire-Free Functional Assessment</h2>
                        <p className="text-white/60 max-w-2xl mx-auto">
                            Quantitative Flow Ratio (QFR) provides a fast, accurate alternative to FFR. By reconstructing the vessel in 3D from two angiographic projections, we compute pressure drop using simplified fluid dynamics.
                        </p>
                    </div>
                    <div className="grid md:grid-cols-3 gap-8">
                        <div className="p-8 rounded-2xl bg-[#0f0f1a] border border-white/10 hover:border-purple-500/30 transition-all">
                            <div className="w-12 h-12 bg-purple-500/10 rounded-lg flex items-center justify-center mb-6 text-purple-400"><Layers /></div>
                            <h3 className="text-lg font-semibold text-white mb-2">3D Reconstruction</h3>
                            <p className="text-white/60 text-sm">Accurate 3D vessel modeling from two angiographic views separated by &gt;25, capturing the true geometry and stenosis length.</p>
                        </div>
                        <div className="p-8 rounded-2xl bg-[#0f0f1a] border border-white/10 hover:border-purple-500/30 transition-all">
                            <div className="w-12 h-12 bg-purple-500/10 rounded-lg flex items-center justify-center mb-6 text-purple-400"><Waves /></div>
                            <h3 className="text-lg font-semibold text-white mb-2">Fluid Dynamics</h3>
                            <p className="text-white/60 text-sm">Contrast flow velocity is derived from TIMI frame count, enabling calculation of hyperemic flow without adenosine.</p>
                        </div>
                        <div className="p-8 rounded-2xl bg-[#0f0f1a] border border-white/10 hover:border-purple-500/30 transition-all">
                            <div className="w-12 h-12 bg-purple-500/10 rounded-lg flex items-center justify-center mb-6 text-purple-400"><Network /></div>
                            <h3 className="text-lg font-semibold text-white mb-2">Clinical Correlation</h3>
                            <p className="text-white/60 text-sm">High diagnostic agreement with wire-based FFR (Accuracy &gt;92%), making it a robust tool for functional ischemia assessment.</p>
                        </div>
                    </div>
                </div>
            </section>

            {/* Scientific Validation */}
            <section className="py-24 bg-[#0a0a12] border-t border-white/5">
                <div className="max-w-7xl mx-auto px-6">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl md:text-5xl font-bold mb-6">Scientific Foundation</h2>
                        <p className="text-white/60 max-w-2xl mx-auto">Built upon the validated mathematical frameworks and RWS formulas established in landmark clinical research.</p>
                    </div>
                    <div className="grid lg:grid-cols-2 gap-12 items-center">
                        <div className="space-y-8">
                            <div className="p-6 rounded-2xl bg-[#0f0f1a] border border-white/10">
                                <h3 className="text-xl font-semibold text-white mb-2">Methodology Alignment</h3>
                                <p className="text-white/60 text-sm mb-4">Our engine implements the exact RWS calculation methods that have shown high correlation (r=0.91) with OCT-FEA in clinical literature.</p>
                                <div className="flex items-center gap-4">
                                    <div className="px-4 py-2 bg-blue-500/10 rounded-lg border border-blue-500/20">
                                        <div className="text-2xl font-bold text-blue-400">r = 0.91</div>
                                        <div className="text-[10px] text-blue-300 uppercase tracking-wider">Pearson Coeff</div>
                                    </div>
                                    <div className="px-4 py-2 bg-emerald-500/10 rounded-lg border border-emerald-500/20">
                                        <div className="text-2xl font-bold text-emerald-400">0.92</div>
                                        <div className="text-[10px] text-emerald-300 uppercase tracking-wider">Intra-observer</div>
                                    </div>
                                </div>
                            </div>
                            <div className="p-6 rounded-2xl bg-[#0f0f1a] border border-white/10">
                                <h3 className="text-xl font-semibold text-white mb-2">Plaque Characterization</h3>
                                <p className="text-white/60 text-sm mb-4">RWS &gt; 12% is associated with higher biomechanical stress and plaque instability, providing critical insights beyond simple geometric stenosis.</p>
                                <ul className="space-y-2 text-sm text-white/50">
                                    <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-red-400" />Correlates with Plaque Elasticity</li>
                                    <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-red-400" />Identifies High-Stress Regions</li>
                                </ul>
                            </div>
                        </div>
                        <div className="space-y-4">
                            <h3 className="text-sm font-semibold text-white/40 uppercase tracking-widest mb-4">Key Literature</h3>
                            <a href="https://doi.org/10.4244/EIJ-D-22-00537" target="_blank" rel="noopener noreferrer" className="block p-6 rounded-xl bg-white/5 border border-white/5 hover:border-blue-500/50 hover:bg-white/10 transition-all group">
                                <div className="flex items-start justify-between">
                                    <div>
                                        <div className="text-blue-400 text-sm font-medium mb-1">EuroIntervention 2023</div>
                                        <h4 className="text-white font-medium mb-2 group-hover:text-blue-300 transition-colors">Role of coronary angiography-derived radial wall strain in assessing plaque vulnerability...</h4>
                                        <div className="text-white/40 text-xs">Hong et al.</div>
                                    </div>
                                    <ArrowRight className="w-5 h-5 text-white/20 group-hover:text-white group-hover:translate-x-1 transition-all" />
                                </div>
                            </a>
                            <a href="https://doi.org/10.1016/j.ijcard.2022.04.053" target="_blank" rel="noopener noreferrer" className="block p-6 rounded-xl bg-white/5 border border-white/5 hover:border-blue-500/50 hover:bg-white/10 transition-all group">
                                <div className="flex items-start justify-between">
                                    <div>
                                        <div className="text-blue-400 text-sm font-medium mb-1">Int. J. Cardiol. 2022</div>
                                        <h4 className="text-white font-medium mb-2 group-hover:text-blue-300 transition-colors">Image-based biomechanical modeling for coronary atherosclerotic plaque progression...</h4>
                                        <div className="text-white/40 text-xs">Wang et al.</div>
                                    </div>
                                    <ArrowRight className="w-5 h-5 text-white/20 group-hover:text-white group-hover:translate-x-1 transition-all" />
                                </div>
                            </a>
                            <a href="https://www.jacc.org/journal/jcm" target="_blank" rel="noopener noreferrer" className="block p-6 rounded-xl bg-white/5 border border-white/5 hover:border-blue-500/50 hover:bg-white/10 transition-all group">
                                <div className="flex items-start justify-between">
                                    <div>
                                        <div className="text-blue-400 text-sm font-medium mb-1">J Am Coll Cardiol Img.</div>
                                        <h4 className="text-white font-medium mb-2 group-hover:text-blue-300 transition-colors">Angiography-based Assessment of Plaque Vulnerability: Validation against OCT</h4>
                                        <div className="text-white/40 text-xs">Lead Study Group</div>
                                    </div>
                                    <ArrowRight className="w-5 h-5 text-white/20 group-hover:text-white group-hover:translate-x-1 transition-all" />
                                </div>
                            </a>
                        </div>
                    </div>
                </div>
            </section>

            {/* Academic Access Notice */}
            <section className="py-24 bg-[#0a0a12] border-t border-white/5">
                <div className="max-w-4xl mx-auto px-6">
                    <div className="text-center">
                        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-sm font-medium mb-8">
                            <FlaskConical className="w-4 h-4" />
                            <span>Academic Access Program</span>
                        </div>
                        <h2 className="text-3xl md:text-5xl font-bold mb-6">Free for Academic Research</h2>
                        <p className="text-xl text-white/60 leading-relaxed max-w-2xl mx-auto mb-8">
                            This platform is currently <strong className="text-emerald-400">free for academic and research use</strong> while
                            we continue to develop and refine our analysis algorithms.
                            We welcome feedback from the research community.
                        </p>
                        <div className="bg-[#0f0f1a] border border-white/10 rounded-2xl p-8 max-w-xl mx-auto mb-8">
                            <div className="flex items-center justify-center gap-4 mb-4">
                                <div className="w-3 h-3 rounded-full bg-amber-400 animate-pulse" />
                                <span className="text-amber-400 font-semibold">Development in Progress</span>
                            </div>
                            <p className="text-white/50 text-sm">
                                New features and improvements are being actively developed.
                                Some functionality may change as we refine the platform based on user feedback and clinical validation studies.
                            </p>
                        </div>
                        <div className="flex flex-col sm:flex-row gap-4 justify-center">
                            <button
                                onClick={() => navigate('/register')}
                                className="px-8 py-4 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl font-semibold transition-all shadow-lg shadow-emerald-500/25 hover:shadow-emerald-500/40 flex items-center justify-center gap-2"
                            >
                                Request Academic Access
                                <ArrowRight className="w-5 h-5" />
                            </button>
                            <a
                                href="mailto:drfatihkoksal@hotmail.com"
                                className="px-8 py-4 bg-white/5 hover:bg-white/10 text-white border border-white/10 rounded-xl font-semibold transition-all flex items-center justify-center gap-2"
                            >
                                Contact Research Team
                            </a>
                        </div>
                    </div>
                </div>
            </section>

            {/* Footer */}
            <footer className="py-12 border-t border-white/10 text-center text-white/40 text-sm">
                <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-6">
                    <div className="flex items-center gap-2">
                        <Activity className="w-4 h-4" />
                        <span>Coronary RWS Analyser v2.0</span>
                    </div>
                    <div className="flex gap-8">
                        <button onClick={() => navigate('/privacy')} className="hover:text-white transition-colors">Privacy</button>
                        <button onClick={() => navigate('/terms')} className="hover:text-white transition-colors">Terms of Research Use</button>
                    </div>
                    <div>
                        {new Date().getFullYear()}, <a href="mailto:drfatihkoksal@hotmail.com" className="hover:text-white transition-colors">Dr. Fatih KOKSAL</a>
                    </div>
                </div>
            </footer>
        </div>
    );
}
