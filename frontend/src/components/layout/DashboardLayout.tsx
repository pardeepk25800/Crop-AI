import type { ReactNode } from 'react';
import Sidebar from './Sidebar';
import { motion, AnimatePresence } from 'framer-motion';

interface Props {
  children: ReactNode;
}

export default function DashboardLayout({ children }: Props) {
  return (
    <div className="flex h-screen bg-background overflow-hidden">
      <Sidebar />
      <main className="flex-1 ml-64 overflow-y-auto relative">
        <header className="sticky top-0 z-40 bg-background/80 backdrop-blur-md px-8 py-4 flex items-center justify-between border-b border-zinc-200/50">
          <div className="flex flex-col">
            <h2 className="text-xl font-bold font-display tracking-tight">Agricultural Intelligence</h2>
            <p className="text-xs text-zinc-500 font-medium uppercase tracking-wider">CropAI Dashboard v1.0</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 bg-white px-3 py-1.5 rounded-full border border-zinc-200 shadow-sm">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              <span className="text-xs font-bold text-zinc-700">API ACTIVE</span>
            </div>
            <div className="w-10 h-10 rounded-full bg-primary-light/10 border border-primary-light/20 flex items-center justify-center overflow-hidden">
              <img src="https://ui-avatars.com/api/?name=Pardeep&background=4CAF50&color=fff" alt="User" />
            </div>
          </div>
        </header>

        <AnimatePresence mode="wait">
          <motion.div
            key="content"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="p-8 pb-20"
          >
            {children}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
