import { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import DashboardLayout from './components/layout/DashboardLayout';
import Home from './pages/Home';
import DiseaseIntelligence from './pages/DiseaseIntelligence';
import YieldIntelligence from './pages/YieldIntelligence';
import History from './pages/History';
import { useAppStore } from './store/useStore';

function App() {
  const { checkHealth } = useAppStore();

  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  return (
    <Router>
      <DashboardLayout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/disease" element={<DiseaseIntelligence />} />
          <Route path="/yield" element={<YieldIntelligence />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </DashboardLayout>
    </Router>
  );
}

export default App;
