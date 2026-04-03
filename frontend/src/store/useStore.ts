import { create } from 'zustand';
import { healthService, diseaseService, yieldService, analyticsService } from '../services/api';

interface AppState {
  isHealthy: boolean;
  isLoading: boolean;
  error: string | null;
  checkHealth: () => Promise<void>;
}

export const useAppStore = create<AppState>((set) => ({
  isHealthy: false,
  isLoading: false,
  error: null,
  checkHealth: async () => {
    set({ isLoading: true });
    try {
      const data = await healthService.check();
      set({ isHealthy: data.status === 'healthy', error: null });
    } catch (err: any) {
      set({ isHealthy: false, error: 'Backend unreachable' });
    } finally {
      set({ isLoading: false });
    }
  },
}));

interface PredictionState {
  diseaseResult: any | null;
  yieldResult: any | null;
  classes: string[];
  crops: string[];
  seasons: string[];
  states: string[];
  fetchMeta: () => Promise<void>;
  predictDisease: (file: File) => Promise<void>;
  predictYield: (data: any) => Promise<void>;
  reset: () => void;
}

export const usePredictionStore = create<PredictionState>((set) => ({
  diseaseResult: null,
  yieldResult: null,
  classes: [],
  crops: [],
  seasons: [],
  states: [],
  fetchMeta: async () => {
    try {
      const [clsData, cropData] = await Promise.all([
        diseaseService.getClasses(),
        yieldService.getCrops(),
      ]);
      set({
        classes: clsData.classes,
        crops: cropData.crops,
        seasons: cropData.seasons,
        states: cropData.states,
      });
    } catch (err) {
      console.error('Failed to fetch metadata', err);
    }
  },
  predictDisease: async (file: File) => {
    set({ diseaseResult: null });
    try {
      const result = await diseaseService.predict(file);
      set({ diseaseResult: result });
    } catch (err) {
      console.error('Disease prediction failed', err);
      throw err;
    }
  },
  predictYield: async (data: any) => {
    set({ yieldResult: null });
    try {
      const result = await yieldService.predict(data);
      set({ yieldResult: result });
    } catch (err) {
      console.error('Yield prediction failed', err);
      throw err;
    }
  },
  reset: () => set({ diseaseResult: null, yieldResult: null }),
}));

interface AnalyticsState {
  stats: any | null;
  history: any[];
  isLoading: boolean;
  fetchAnalytics: () => Promise<void>;
}

export const useAnalyticsStore = create<AnalyticsState>((set) => ({
  stats: null,
  history: [],
  isLoading: false,
  fetchAnalytics: async () => {
    set({ isLoading: true });
    try {
      const [statsData, historyData] = await Promise.all([
        analyticsService.getStats(),
        analyticsService.getHistory(),
      ]);
      set({ stats: statsData, history: historyData.history });
    } catch (err) {
      console.error('Failed to fetch analytics', err);
    } finally {
      set({ isLoading: false });
    }
  },
}));
