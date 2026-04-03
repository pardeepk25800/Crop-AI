import { useNavigate, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { LayoutDashboard, Leaf, TrendingUp, History, Settings, LogOut } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const navItems = [
  { id: 'dashboard', icon: LayoutDashboard, label: 'Dashboard', path: '/' },
  { id: 'disease', icon: Leaf, label: 'Disease Intel', path: '/disease' },
  { id: 'yield', icon: TrendingUp, label: 'Yield Intel', path: '/yield' },
  { id: 'history', icon: History, label: 'History', path: '/history' },
];

export default function Sidebar() {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <div className="w-64 h-full bg-[#1A3A1A] text-white flex flex-col p-6 fixed left-0 top-0 shadow-2xl z-50">
      <div className="flex items-center gap-3 mb-10 px-2">
        <div className="w-10 h-10 bg-primary-light rounded-xl flex items-center justify-center shadow-lg">
          <Leaf className="text-white fill-white" size={24} />
        </div>
        <h1 className="text-2xl font-bold tracking-tighter font-display">CropAI</h1>
      </div>

      <nav className="flex-1 space-y-2">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <motion.button
              key={item.id}
              whileHover={{ x: 4 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => navigate(item.path)}
              className={cn(
                "w-full flex items-center gap-4 px-4 py-3 rounded-xl transition-all duration-200 group relative",
                isActive 
                  ? "bg-white/10 text-primary-light border border-white/10" 
                  : "text-zinc-400 hover:text-white hover:bg-white/5"
              )}
            >
              <item.icon size={20} className={cn(isActive && "text-primary-light")} />
              <span className="font-semibold text-sm">{item.label}</span>
              {isActive && (
                <motion.div
                  layoutId="active-pill"
                  className="absolute left-0 w-1 h-6 bg-primary-light rounded-r-full"
                />
              )}
            </motion.button>
          );
        })}
      </nav>

      <div className="pt-6 border-t border-white/5 space-y-2">
        <button className="w-full flex items-center gap-4 px-4 py-3 rounded-xl text-zinc-400 hover:text-white hover:bg-white/5 transition-all group">
          <Settings size={20} />
          <span className="font-semibold text-sm">Settings</span>
        </button>
        <button className="w-full flex items-center gap-4 px-4 py-3 rounded-xl text-red-400/80 hover:text-red-400 hover:bg-red-500/10 transition-all group">
          <LogOut size={20} />
          <span className="font-semibold text-sm">Logout</span>
        </button>
      </div>
    </div>
  );
}
