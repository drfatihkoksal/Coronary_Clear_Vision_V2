import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Shield, Lock, Eye, Database, UserCheck } from 'lucide-react';

export function PrivacyPage() {
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
                    <h1 className="text-4xl font-bold mb-4">Privacy Policy</h1>
                    <p className="text-xl text-white/60">Last updated: January 2026</p>
                </header>

                <section className="bg-emerald-500/10 border border-emerald-500/20 rounded-2xl p-8 mb-12 relative overflow-hidden">
                    <div className="absolute top-0 right-0 p-8 opacity-10">
                        <Shield className="w-32 h-32 text-emerald-500" />
                    </div>
                    <div className="relative z-10">
                        <div className="flex items-center gap-3 mb-4 text-emerald-400">
                            <Shield className="w-6 h-6" />
                            <h2 className="text-lg font-bold uppercase tracking-wider">Your Data, Your Control</h2>
                        </div>
                        <p className="text-lg text-emerald-100/90 leading-relaxed font-medium">
                            Coronary RWS Analyser is designed with privacy-first principles. We implement end-to-end encryption,
                            minimal data collection, and give you full control over your uploaded data.
                        </p>
                    </div>
                </section>

                <div className="space-y-12 text-white/80 leading-relaxed">
                    <section>
                        <div className="flex items-center gap-3 mb-4 text-blue-400">
                            <Database className="w-6 h-6" />
                            <h2 className="text-2xl font-semibold text-white">1. Data We Collect</h2>
                        </div>
                        <div className="space-y-4">
                            <p>We collect only the data necessary to provide our research services:</p>
                            <ul className="list-disc pl-6 space-y-2 text-white/60 marker:text-blue-500">
                                <li><strong>Account Information:</strong> Email address, name (optional), and encrypted password hash.</li>
                                <li><strong>Uploaded DICOM Data:</strong> Coronary angiography files you upload for analysis. These are temporarily stored during your session.</li>
                                <li><strong>Analysis Results:</strong> Segmentation masks, RWS calculations, and export files you generate.</li>
                                <li><strong>Usage Analytics:</strong> Anonymized usage patterns (no personal identifiers) for service improvement.</li>
                            </ul>
                        </div>
                    </section>

                    <section>
                        <div className="flex items-center gap-3 mb-4 text-blue-400">
                            <Lock className="w-6 h-6" />
                            <h2 className="text-2xl font-semibold text-white">2. Data Storage & Security</h2>
                        </div>
                        <div className="space-y-4">
                            <ul className="list-disc pl-6 space-y-2 text-white/60 marker:text-blue-500">
                                <li><strong>Encryption:</strong> All data is encrypted in transit (TLS 1.3) and at rest (AES-256).</li>
                                <li><strong>Session-Based Storage:</strong> DICOM files are processed in-memory when possible and deleted after session expiry.</li>
                                <li><strong>Password Security:</strong> We use bcrypt with salt for password hashing. We never store plain-text passwords.</li>
                                <li><strong>Access Control:</strong> Strict role-based access ensures only you can view your data.</li>
                            </ul>
                        </div>
                    </section>

                    <section>
                        <div className="flex items-center gap-3 mb-4 text-blue-400">
                            <Eye className="w-6 h-6" />
                            <h2 className="text-2xl font-semibold text-white">3. How We Use Your Data</h2>
                        </div>
                        <div className="space-y-4">
                            <ul className="list-disc pl-6 space-y-2 text-white/60 marker:text-blue-500">
                                <li>To provide vessel segmentation, RWS calculation, and reporting services.</li>
                                <li>To authenticate your identity and secure your account.</li>
                                <li>To improve our AI models (only with explicit opt-in consent for anonymized data contribution).</li>
                                <li>To communicate important service updates or security notifications.</li>
                            </ul>
                            <p className="mt-4 text-amber-400/80 bg-amber-500/10 px-4 py-3 rounded-lg border border-amber-500/20">
                                <strong>We never sell your data to third parties.</strong> Your medical imaging data is not used for advertising or shared with any external entities.
                            </p>
                        </div>
                    </section>

                    <section>
                        <div className="flex items-center gap-3 mb-4 text-blue-400">
                            <UserCheck className="w-6 h-6" />
                            <h2 className="text-2xl font-semibold text-white">4. Your Rights</h2>
                        </div>
                        <div className="space-y-4">
                            <p>You have the right to:</p>
                            <ul className="list-disc pl-6 space-y-2 text-white/60 marker:text-blue-500">
                                <li><strong>Access:</strong> Request a copy of all data we hold about you.</li>
                                <li><strong>Rectification:</strong> Correct inaccurate account information.</li>
                                <li><strong>Erasure:</strong> Request deletion of your account and all associated data.</li>
                                <li><strong>Portability:</strong> Export your analysis results in standard formats (JSON, CSV).</li>
                                <li><strong>Withdrawal:</strong> Revoke consent for optional data processing at any time.</li>
                            </ul>
                            <p className="mt-4">
                                To exercise these rights, contact us at <a href="mailto:drfatihkoksal@hotmail.com" className="text-blue-400 underline">drfatihkoksal@hotmail.com</a>.
                            </p>
                        </div>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold text-white mb-4">5. Data Retention</h2>
                        <ul className="list-disc pl-6 space-y-2 text-white/60 marker:text-blue-500">
                            <li><strong>Active accounts:</strong> Data retained while account is active.</li>
                            <li><strong>Session data (DICOM):</strong> Deleted within 24 hours of session end.</li>
                            <li><strong>Deleted accounts:</strong> All user data permanently erased within 30 days.</li>
                            <li><strong>Anonymized analytics:</strong> Retained indefinitely for research purposes.</li>
                        </ul>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold text-white mb-4">6. Cookies & Tracking</h2>
                        <p>
                            We use only essential cookies for authentication (JWT session tokens). We do not use third-party
                            tracking cookies, Google Analytics, or any advertising trackers. Your browsing behavior is not monitored.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold text-white mb-4">7. Contact</h2>
                        <p>
                            For questions about this Privacy Policy or data handling practices, please contact:<br />
                            <a href="mailto:drfatihkoksal@hotmail.com" className="text-blue-400 underline">drfatihkoksal@hotmail.com</a>
                        </p>
                    </section>
                </div>

                <div className="mt-16 pt-8 border-t border-white/10 text-center">
                    <button
                        onClick={() => navigate('/register')}
                        className="px-8 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-white font-medium transition-all"
                    >
                        Continue to Registration
                    </button>
                </div>
            </main>
        </div>
    );
}
