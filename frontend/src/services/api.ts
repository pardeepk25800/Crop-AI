import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const diseaseService = {
  getClasses: async () => {
    const response = await api.get('/classes');
    return response.data;
  },
  predict: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post('/predict/disease', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
};

export const yieldService = {
  getCrops: async () => {
    const response = await api.get('/crops');
    return response.data;
  },
  predict: async (data: any) => {
    const response = await api.post('/predict/yield', data);
    return response.data;
  },
};

export const healthService = {
  check: async () => {
    const response = await api.get('/health');
    return response.data;
  },
};

export const analyticsService = {
  getStats: async () => {
    const response = await api.get('/stats');
    return response.data;
  },
  getHistory: async (limit: number = 50) => {
    const response = await api.get(`/history?limit=${limit}`);
    return response.data;
  },
};

export default api;
