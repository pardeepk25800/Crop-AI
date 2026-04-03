import { useEffect } from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, Cloudy, Droplets, Thermometer, FlaskConical, Bug } from 'lucide-react';
import { usePredictionStore } from '../store/useStore';

export default function YieldIntelligence() {
  const fetchMeta = usePredictionStore((state: any) => state.fetchMeta);
  const predictYield = usePredictionStore((state: any) => state.predictYield);
  const yieldResult = usePredictionStore((state: any) => state.yieldResult);
  const crops = usePredictionStore((state: any) => state.crops);
  const seasons = usePredictionStore((state: any) => state.seasons);
  const states = usePredictionStore((state: any) => state.states);

  useEffect(() => {
    fetchMeta();
  }, [fetchMeta]);

  const handleSubmit = async (e: any) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());
    
    // Convert numeric fields
    const numericFields = ['area', 'rainfall', 'temperature', 'fertilizer', 'pesticide'];
    const payload: any = { ...data };
    numericFields.forEach(field => {
      if (payload[field]) payload[field] = parseFloat(payload[field] as string);
    });

    await predictYield(payload);
  };

  return (
    <div className="max-w-6xl mx-auto space-y-12 pb-20">
      <div className="flex justify-between items-end border-b border-zinc-200 pb-8">
        <div className="space-y-2">
          <span className="text-[10px] font-black uppercase tracking-[0.2em] text-primary">Intelligence Engine</span>
          <h1 className="text-6xl font-extrabold font-display tracking-tight leading-none">CROP YIELD<br/>FORCASTING</h1>
        </div>
        <p className="max-w-[300px] text-sm font-medium text-zinc-500 text-right leading-relaxed mb-1">
          Utilizing multi-ensemble ML models to predict regional crop performance based on hyper-local environmental variables.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-12">
        <form onSubmit={handleSubmit} className="lg:col-span-4 space-y-10">
          <div className="space-y-6">
            <h3 className="text-xs font-black uppercase tracking-widest text-zinc-400 border-l-2 border-primary-light pl-3">Field Parameters</h3>
            
            <div className="space-y-4">
              <div className="space-y-1.5">
                <label className="text-[10px] uppercase font-bold text-zinc-500">Crop Variant</label>
                <select name="crop" className="w-full bg-white border border-zinc-200 rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-primary-light/20 transition-all font-bold text-zinc-700">
                  {crops.map((c: string) => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-[10px] uppercase font-bold text-zinc-500">Season</label>
                  <select name="season" className="w-full bg-white border border-zinc-200 rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-primary-light/20 transition-all font-bold text-zinc-700">
                    {seasons.map((s: string) => <option key={s} value={s}>{s}</option>)}
                  </select>
                </div>
                <div className="space-y-1.5">
                  <label className="text-[10px] uppercase font-bold text-zinc-500">Field Area (Ha)</label>
                  <input name="area" type="number" step="0.1" defaultValue="5.0" className="w-full bg-white border border-zinc-200 rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-primary-light/20 transition-all font-bold text-zinc-700" />
                </div>
              </div>
              <div className="space-y-1.5">
                <label className="text-[10px] uppercase font-bold text-zinc-500">Regional State</label>
                <select name="state" className="w-full bg-white border border-zinc-200 rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-primary-light/20 transition-all font-bold text-zinc-700">
                  {states.map((s: string) => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <h3 className="text-xs font-black uppercase tracking-widest text-zinc-400 border-l-2 border-primary-light pl-3">Environmental Variables</h3>
            <div className="grid grid-cols-2 gap-6">
              <div className="space-y-1.5">
                <label className="text-[10px] uppercase font-bold text-zinc-500 flex items-center gap-1.5">
                  <Droplets size={12}/> Rainfall (mm)
                </label>
                <input name="rainfall" type="number" defaultValue="1200" className="w-full bg-white border border-zinc-200 rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-primary-light/20 transition-all font-bold text-zinc-700" />
              </div>
              <div className="space-y-1.5">
                <label className="text-[10px] uppercase font-bold text-zinc-500 flex items-center gap-1.5">
                  <Thermometer size={12}/> Temp (°C)
                </label>
                <input name="temperature" type="number" step="0.1" defaultValue="28" className="w-full bg-white border border-zinc-200 rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-primary-light/20 transition-all font-bold text-zinc-700" />
              </div>
              <div className="space-y-1.5">
                <label className="text-[10px] uppercase font-bold text-zinc-500 flex items-center gap-1.5">
                  <FlaskConical size={12}/> Fertilizer (kg)
                </label>
                <input name="fertilizer" type="number" defaultValue="150" className="w-full bg-white border border-zinc-200 rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-primary-light/20 transition-all font-bold text-zinc-700" />
              </div>
              <div className="space-y-1.5">
                <label className="text-[10px] uppercase font-bold text-zinc-500 flex items-center gap-1.5">
                  <Bug size={12}/> Pesticide (kg)
                </label>
                <input name="pesticide" type="number" step="0.1" defaultValue="2.0" className="w-full bg-white border border-zinc-200 rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-primary-light/20 transition-all font-bold text-zinc-700" />
              </div>
            </div>
          </div>

          <button type="submit" className="w-full bg-zinc-900 text-white py-5 rounded-2xl font-black uppercase tracking-widest hover:bg-black hover:scale-[1.02] active:scale-[0.98] transition-all shadow-xl shadow-zinc-200">
            Generate Forecast
          </button>
        </form>

        <div className="lg:col-span-8 space-y-8">
          {yieldResult ? (
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="glass rounded-[50px] p-12 border-primary/20 relative overflow-hidden bg-primary/5"
            >
              <div className="absolute -top-10 -right-10 w-64 h-64 bg-primary-light/10 rounded-full blur-3xl" />
              
              <div className="flex justify-between items-start mb-12">
                <div>
                  <h2 className="text-2xl font-bold font-display text-primary uppercase tracking-tight">Yield Estimates</h2>
                  <p className="text-xs text-zinc-500 font-bold uppercase tracking-widest mt-1">{yieldResult.crop} • {yieldResult.season}</p>
                </div>
                <div className="px-6 py-2 bg-white rounded-full border border-primary/10 shadow-sm">
                  <span className="text-[10px] font-black text-primary uppercase tracking-widest leading-none">Grade: {yieldResult.yield_grade}</span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                <div className="space-y-10">
                  <div>
                    <p className="text-[10px] uppercase font-black tracking-widest text-zinc-400 mb-2">Predicted Efficiency</p>
                    <div className="flex items-baseline gap-2">
                      <span className="text-7xl font-black text-zinc-800 tracking-tighter">{yieldResult.yield_per_ha}</span>
                      <span className="text-lg font-bold text-zinc-400 uppercase">kg/ha</span>
                    </div>
                  </div>
                  <div>
                    <p className="text-[10px] uppercase font-black tracking-widest text-zinc-400 mb-2">Total Estimated Harvest</p>
                    <div className="flex items-baseline gap-2">
                      <span className="text-5xl font-black text-zinc-800 tracking-tighter">{yieldResult.total_yield}</span>
                      <span className="text-base font-bold text-zinc-400 uppercase">kg total</span>
                    </div>
                  </div>
                </div>

                <div className="bg-white/60 backdrop-blur-sm rounded-[32px] p-8 border border-white shadow-xl shadow-primary/5">
                  <h4 className="text-xs font-black uppercase tracking-widest text-zinc-500 mb-6 flex items-center gap-2">
                    <Cloudy size={14} className="text-primary" /> Smart Recommendations
                  </h4>
                  <ul className="space-y-5">
                    {yieldResult.recommendations.map((r: string, i: number) => (
                      <li key={i} className="flex gap-4 items-start group">
                        <div className="w-2 h-2 rounded-full bg-primary-light mt-1.5 transition-all group-hover:scale-150" />
                        <p className="text-sm font-medium text-zinc-600 line-height-relaxed">{r}</p>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </motion.div>
          ) : (
            <div className="h-full min-h-[500px] glass rounded-[50px] border-zinc-100 border-dashed border-2 flex flex-col items-center justify-center text-center p-20">
              <TrendingUp className="text-zinc-200 mb-6" size={80} strokeWidth={1} />
              <h3 className="text-2xl font-bold font-display text-zinc-300">Awaiting Simulation Parameters</h3>
              <p className="text-sm text-zinc-400 max-w-[320px] mt-2 leading-relaxed">Fill in the field data and environmental variables on the left to generate a yield intelligence forecast. </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
