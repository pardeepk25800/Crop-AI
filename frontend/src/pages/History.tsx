import { useEffect } from 'react';
import { motion } from 'framer-motion';
import { Download, Search, Filter, Calendar } from 'lucide-react';
import { useAnalyticsStore } from '../store/useStore';

export default function History() {
  const fetchAnalytics = useAnalyticsStore((state: any) => state.fetchAnalytics);
  const historyData = useAnalyticsStore((state: any) => state.history);

  useEffect(() => {
    fetchAnalytics();
  }, [fetchAnalytics]);
  return (
    <div className="space-y-8">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-4xl font-extrabold font-display uppercase tracking-tight">Activity Log</h1>
          <p className="text-zinc-500 font-medium">Review and export your historical agricultural intelligence data.</p>
        </div>
        <div className="flex gap-3">
          <button className="flex items-center gap-2 bg-white px-5 py-2.5 rounded-xl border border-zinc-200 shadow-sm text-sm font-bold text-zinc-700 hover:bg-zinc-50">
            <Filter size={18} /> Filter
          </button>
          <button className="flex items-center gap-2 bg-primary px-5 py-2.5 rounded-xl shadow-lg text-sm font-bold text-white hover:bg-primary-dark">
            <Download size={18} /> Export Data
          </button>
        </div>
      </div>

      <div className="glass rounded-[32px] overflow-hidden">
        <div className="p-6 bg-white/50 border-b border-zinc-100 flex flex-col md:flex-row gap-4 items-center justify-between">
          <div className="relative w-full md:w-96">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-zinc-400" size={18} />
            <input 
              type="text" 
              placeholder="Search by crop, disease or result..." 
              className="w-full pl-12 pr-4 py-3 bg-white border border-zinc-200 rounded-xl outline-none focus:ring-2 focus:ring-primary/20 font-medium text-sm transition-all"
            />
          </div>
          <div className="flex items-center gap-2 text-zinc-400 text-sm font-bold">
            <Calendar size={18} />
            <span>March 2026</span>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-zinc-50/50">
                <th className="px-8 py-4 text-[10px] font-black uppercase tracking-widest text-zinc-400">Analysis Type</th>
                <th className="px-8 py-4 text-[10px] font-black uppercase tracking-widest text-zinc-400">Crop</th>
                <th className="px-8 py-4 text-[10px] font-black uppercase tracking-widest text-zinc-400">Primary Result</th>
                <th className="px-8 py-4 text-[10px] font-black uppercase tracking-widest text-zinc-400">Confidence/Grade</th>
                <th className="px-8 py-4 text-[10px] font-black uppercase tracking-widest text-zinc-400">Timestamp</th>
                <th className="px-8 py-4 text-[10px] font-black uppercase tracking-widest text-zinc-400">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100">
              {historyData.map((item: any, i: number) => (
                <motion.tr 
                  key={item.id}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="hover:bg-zinc-50/50 transition-colors group"
                >
                  <td className="px-8 py-5">
                    <span className={`px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest ${
                      item.analysis_type === 'Disease' ? 'bg-indigo-50 text-indigo-600' : 'bg-emerald-50 text-emerald-600'
                    }`}>
                      {item.analysis_type}
                    </span>
                  </td>
                  <td className="px-8 py-5 font-bold text-zinc-800">{item.crop || 'Unknown'}</td>
                  <td className="px-8 py-5">
                    <span className={`px-4 py-1.5 rounded-xl font-bold text-sm ${
                      item.analysis_type === 'Disease' 
                        ? (item.is_healthy ? 'bg-green-50 text-green-600' : 'bg-red-50 text-red-600') 
                        : 'bg-blue-50 text-blue-600'
                    }`}>
                      {item.analysis_type === 'Disease' ? item.predicted_class : `${item.yield_per_ha.toFixed(0)} kg/ha`}
                    </span>
                  </td>
                  <td className="px-8 py-5 font-bold text-zinc-600">
                    {item.analysis_type === 'Disease' ? `${item.confidence.toFixed(1)}%` : item.yield_grade}
                  </td>
                  <td className="px-8 py-5 text-xs font-medium text-zinc-400">
                    {new Date(item.timestamp).toLocaleString(undefined, {
                        month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
                    })}
                  </td>
                  <td className="px-8 py-5">
                    <button className="text-zinc-400 hover:text-primary transition-colors">
                      <Download size={18} />
                    </button>
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
