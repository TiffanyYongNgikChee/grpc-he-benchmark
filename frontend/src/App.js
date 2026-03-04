import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import PredictPage from "./pages/PredictPage";
import BenchmarkPage from "./pages/BenchmarkPage";
import AboutPage from "./pages/AboutPage";

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-slate-900 text-slate-200">
        <Navbar />
        <Routes>
          <Route path="/" element={<PredictPage />} />
          <Route path="/benchmark" element={<BenchmarkPage />} />
          <Route path="/about" element={<AboutPage />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
