import { useNavigate } from 'react-router-dom';
import { ArrowLeft, ShieldAlert, FileText, FlaskConical } from 'lucide-react';

export function TermsPage() {
    const navigate = useNavigate();

    return (
        <div className="min-h-screen bg-[#0f0f1a] text-white selection:bg-blue-500/30 font-sans">
            <nav className="border-b border-white/10 bg-[#0f0f1a]/80 backdrop-blur-md sticky top-0 z-50">
                <div className="max-w-4xl mx-auto px-6 h-20 flex items-center">
                    <button
                        onClick={() => navigate('/')}
                        className="text-white/60 hover:text-white flex items-center gap-2 transition-colors"
                    >
                        <ArrowLeft className="w-5 h-5" />
                        Back to Home
                    </button>
                </div>
            </nav>

            <main className="max-w-4xl mx-auto px-6 py-12">
                <header className="mb-12">
                    <h1 className="text-4xl font-bold mb-4">Terms of Use & Disclaimer</h1>
                    <p className="text-xl text-white/60">Last updated: January 2026</p>
                </header>

                <section className="bg-amber-500/10 border border-amber-500/20 rounded-2xl p-8 mb-12 relative overflow-hidden">
                    <div className="absolute top-0 right-0 p-8 opacity-10">
                        <ShieldAlert className="w-32 h-32 text-amber-500" />
                    </div>
                    <div className="relative z-10">
                        <div className="flex items-center gap-3 mb-4 text-amber-400">
                            <ShieldAlert className="w-6 h-6" />
                            <h2 className="text-lg font-bold uppercase tracking-wider">Research Use Only Statement</h2>
                        </div>
                        <p className="text-lg text-amber-100/90 leading-relaxed font-medium">
                            The Coronary RWS Analyser software is a research tool tailored for academic and investigational use.
                            It has <span className="underline decoration-amber-500/50 decoration-2 underline-offset-2">not</span> been approved, cleared, or licensed by the FDA, EMA, or any other regulatory body for clinical diagnostic use.
                        </p>
                        <ul className="mt-6 space-y-3 text-amber-100/80">
                            <li className="flex items-start gap-3">
                                <div className="mt-1.5 w-1.5 h-1.5 rounded-full bg-amber-400 shrink-0" />
                                <span>Do not use this software for primary diagnosis or patient management decisions.</span>
                            </li>
                            <li className="flex items-start gap-3">
                                <div className="mt-1.5 w-1.5 h-1.5 rounded-full bg-amber-400 shrink-0" />
                                <span>All outputs (RWS values, segmentation masks, risk scores) are for informational and research purposes only.</span>
                            </li>
                        </ul>
                    </div>
                </section>

                <div className="space-y-12 text-white/80 leading-relaxed">
                    <section>
                        <div className="flex items-center gap-3 mb-4 text-blue-400">
                            <FlaskConical className="w-6 h-6" />
                            <h2 className="text-2xl font-semibold text-white">1. Intended Audience</h2>
                        </div>
                        <p>
                            This platform is designed strictly for researchers, data scientists, and academic institutions conducting retrospective studies or validation trials in the field of coronary physiology and biomechanics. By creating an account, you certify that you are accessing this tool for research or educational purposes.
                        </p>
                    </section>

                    <section>
                        <div className="flex items-center gap-3 mb-4 text-blue-400">
                            <FileText className="w-6 h-6" />
                            <h2 className="text-2xl font-semibold text-white">2. No Medical Advice</h2>
                        </div>
                        <p>
                            The content, data, and visualizations provided by the Coronary RWS Analyser do not constitute medical advice, diagnosis, or treatment recommendations. The "Vulnerable Plaque" and "Risk" classifications are based on theoretical models (Hong et al., EuroIntervention 2023) and should be interpreted as experimental metrics, not definitive clinical findings.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold text-white mb-4">3. Data Privacy & HIPAA</h2>
                        <p className="mb-4">
                            While we implement industry-standard encryption and security practices, you act as the Data Controller for any DICOM data you upload. You warrant that:
                        </p>
                        <ul className="list-disc pl-6 space-y-2 text-white/60 marker:text-blue-500">
                            <li>All uploaded data has been de-identified / anonymized in accordance with HIPAA (Safe Harbor method) or GDPR requirements locally.</li>
                            <li>You have obtained necessary IRB/Ethics Committee approvals for your research analysis.</li>
                            <li>You will not upload Protected Health Information (PHI) unless a specific Data Processing Agreement (DPA) is signed with us.</li>
                        </ul>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold text-white mb-4">4. Limitation of Liability</h2>
                        <p>
                            To the maximum extent permitted by law, the developers and maintainers of Coronary RWS Analyser shall not be liable for any direct, indirect, incidental, or consequential damages arising from the use of this software, including but not limited to incorrect data processing, service interruptions, or reliance on analysis results for medical decisions.
                        </p>
                    </section>
                </div>

                <div className="mt-16 pt-8 border-t border-white/10 text-center">
                    <button
                        onClick={() => navigate('/register')}
                        className="px-8 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-white font-medium transition-all"
                    >
                        I Understand & Agree to these Terms
                    </button>
                </div>
            </main>
        </div>
    );
}
