import { useEffect } from 'react';
import { motion } from 'framer-motion';
import { Leaf, TrendingUp, History, Activity } from 'lucide-react';
import { useAnalyticsStore } from '../store/useStore';

export default function Home() {
  const fetchAnalytics = useAnalyticsStore((state: any) => state.fetchAnalytics);
  const stats = useAnalyticsStore((state: any) => state.stats);
  const history = useAnalyticsStore((state: any) => state.history);

  useEffect(() => {
    fetchAnalytics();
  }, [fetchAnalytics]);

  const totalScans = stats ? stats.total_disease_predictions : 0;
  const totalForecasts = stats ? stats.total_yield_predictions : 0;
  const totalHistory = totalScans + totalForecasts;
  const avgConf = stats ? stats.avg_disease_confidence : 0;

  const displayStats = [
    { label: 'Disease Scans', value: totalScans.toString(), icon: Leaf, color: 'text-green-600', bg: 'bg-green-100' },
    { label: 'Yield Forecasts', value: totalForecasts.toString(), icon: TrendingUp, color: 'text-blue-600', bg: 'bg-blue-100' },
    { label: 'Total Analyses', value: totalHistory.toString(), icon: History, color: 'text-amber-600', bg: 'bg-amber-100' },
    { label: 'Avg. Confidence', value: `${avgConf.toFixed(1)}%`, icon: Activity, color: 'text-purple-600', bg: 'bg-purple-100' },
  ];

  const recentActivity = history ? history.slice(0, 4) : [];

  return (
    <div className="space-y-8">
      <div className="flex flex-col gap-1">
        <h1 className="text-4xl font-extrabold font-display">Welcome Back, Pardeep</h1>
        <p className="text-zinc-500 font-medium">Here's what's happening on your farms today.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {displayStats.map((stat, i) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            className="glass p-6 rounded-2xl flex items-center gap-5 shadow-sm hover:shadow-md transition-shadow"
          >
            <div className={`w-12 h-12 ${stat.bg} ${stat.color} rounded-xl flex items-center justify-center`}>
              <stat.icon size={24} />
            </div>
            <div>
              <p className="text-[10px] uppercase tracking-widest font-bold text-zinc-400">{stat.label}</p>
              <p className="text-2xl font-bold text-zinc-800 tracking-tighter">{stat.value}</p>
            </div>
          </motion.div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 glass p-8 rounded-3xl min-h-[400px]">
          <h3 className="text-xl font-bold font-display mb-6">Yield Trends</h3>
          <div className="w-full h-full flex items-center justify-center text-zinc-300 italic">
            Chart component will be integrated here
          </div>
        </div>
        <div className="glass p-8 rounded-3xl">
          <h3 className="text-xl font-bold font-display mb-6">Recent Activity</h3>
          <div className="space-y-6">
            {recentActivity.length > 0 ? (
              recentActivity.map((item: any, n: number) => (
                <div key={n} className="flex gap-4 items-start">
                  <div className="w-8 h-8 rounded-full bg-zinc-100 flex items-center justify-center mt-1">
                    {item.analysis_type === 'Disease' ? (
                      <Leaf size={14} className="text-green-500" />
                    ) : (
                      <TrendingUp size={14} className="text-blue-500" />
                    )}
                  </div>
                  <div>
                    <p className="text-sm font-bold text-zinc-800">
                      {item.analysis_type === 'Disease' ? `${item.predicted_class} Scan` : `${item.crop} Yield Forecast`}
                    </p>
                    <p className="text-xs text-zinc-500">
                      {new Date(item.timestamp).toLocaleString(undefined, {
                        month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
                      })}
                      {item.analysis_type === 'Disease' ? ` • ${item.confidence.toFixed(1)}% Conf` : ` • ${item.yield_per_ha.toFixed(0)} kg/ha`}
                    </p>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-sm font-medium text-zinc-400 italic">No recent activity found.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
