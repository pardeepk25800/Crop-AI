import { useState } from 'react';
import { motion } from 'framer-motion';
import { Upload, Camera, Search, AlertCircle, CheckCircle2 } from 'lucide-react';
import { usePredictionStore } from '../store/useStore';

export default function DiseaseIntelligence() {
  const [dragActive, setDragActive] = useState(false);
  const predictDisease = usePredictionStore((state: any) => state.predictDisease);
  const diseaseResult = usePredictionStore((state: any) => state.diseaseResult);

  const handleFileUpload = async (e: any) => {
    const file = e.target.files?.[0];
    if (file) {
      await predictDisease(file);
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-10">
      <div className="text-center space-y-3">
        <h1 className="text-5xl font-extrabold font-display tracking-tight uppercase">Disease Intelligence</h1>
        <p className="text-zinc-500 max-w-xl mx-auto font-medium">
          Upload a clear photo of the infected crop leaf for precise AI-powered analysis and treatment recommendations.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-start">
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="space-y-6"
        >
          <div 
            className={`relative h-[400px] rounded-[40px] border-2 border-dashed transition-all duration-300 flex flex-col items-center justify-center p-10 cursor-pointer ${
              dragActive ? 'border-primary-light bg-primary-light/5' : 'border-zinc-300 hover:border-primary-light/50 bg-white/50 backdrop-blur-sm'
            }`}
            onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={() => setDragActive(false)}
            onDrop={(e) => { e.preventDefault(); setDragActive(false); }}
          >
            <input type="file" className="absolute inset-0 opacity-0 cursor-pointer" onChange={handleFileUpload} accept="image/*" />
            
            <div className="w-20 h-20 bg-primary-light/10 rounded-full flex items-center justify-center mb-6">
              <Upload className="text-primary-light" size={32} />
            </div>
            <h3 className="text-xl font-bold text-zinc-800 mb-2">Drag and drop leaf image</h3>
            <p className="text-zinc-400 text-sm font-medium">Supports JPG, PNG, WEBP (Max 10MB)</p>
            
            <div className="mt-8 flex gap-4">
              <button className="flex items-center gap-2 bg-white px-5 py-2.5 rounded-full border border-zinc-200 shadow-sm text-sm font-bold text-zinc-700 hover:bg-zinc-50 transition-colors">
                <Camera size={18} /> Take Photo
              </button>
              <button className="flex items-center gap-2 bg-zinc-900 px-5 py-2.5 rounded-full shadow-lg text-sm font-bold text-white hover:bg-black transition-colors">
                <Search size={18} /> Browse Files
              </button>
            </div>
          </div>

          <div className="glass p-6 rounded-3xl flex items-center gap-4 border-amber-200 bg-amber-50/30">
            <AlertCircle className="text-amber-600 shrink-0" size={24} />
            <p className="text-xs font-semibold text-amber-800 leading-relaxed">
              For best results, ensure the leaf is well-lit and the main infected area is in focus. Avoid blurry or low-resolution images.
            </p>
          </div>
        </motion.div>

        <motion.div 
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="space-y-6 lg:sticky lg:top-24"
        >
          {diseaseResult ? (
            <div className="glass rounded-[40px] p-8 overflow-hidden relative border-primary/20 bg-white/80">
              <div className="flex items-start justify-between mb-8">
                <div>
                  <span className={`px-4 py-1.5 rounded-full text-[10px] font-black uppercase tracking-widest ${
                    diseaseResult.is_healthy ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                  }`}>
                    {diseaseResult.is_healthy ? 'HEALTHY LEAF' : 'DISEASE DETECTED'}
                  </span>
                  <h2 className="text-3xl font-extrabold font-display mt-3 leading-tight">{diseaseResult.predicted_class}</h2>
                </div>
                <div className="text-right">
                  <p className="text-[10px] uppercase font-black tracking-widest text-zinc-400 mb-1">Confidence</p>
                  <p className="text-4xl font-black text-primary tracking-tighter">{(diseaseResult.confidence).toFixed(1)}%</p>
                </div>
              </div>

              <div className="space-y-8">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-white/50 p-4 rounded-2xl border border-zinc-100">
                    <p className="text-[10px] uppercase font-bold text-zinc-400 mb-1">Severity</p>
                    <p className="font-bold text-zinc-800">{diseaseResult.severity || 'N/A'}</p>
                  </div>
                  <div className="bg-white/50 p-4 rounded-2xl border border-zinc-100">
                    <p className="text-[10px] uppercase font-bold text-zinc-400 mb-1">Spread Risk</p>
                    <p className="font-bold text-zinc-800">{diseaseResult.spread_risk || 'N/A'}</p>
                  </div>
                </div>

                <div className="space-y-3">
                  <h4 className="text-xs uppercase font-black tracking-widest text-primary flex items-center gap-2">
                    <CheckCircle2 size={14} /> Recommended Treatment
                  </h4>
                  <div className="bg-primary/5 p-6 rounded-3xl border border-primary/10">
                    <p className="text-sm font-medium text-zinc-700 leading-relaxed">
                      {diseaseResult.treatment}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="h-[400px] rounded-[40px] border-2 border-zinc-100 flex flex-col items-center justify-center p-12 text-center bg-zinc-50/50">
              <Search className="text-zinc-200 mb-4" size={64} strokeWidth={1} />
              <p className="text-zinc-400 font-bold mb-2">Awaiting Analysis</p>
              <p className="text-xs text-zinc-400 max-w-[240px]">Result and diagnostic details will appear here once an image is uploaded.</p>
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );
}
