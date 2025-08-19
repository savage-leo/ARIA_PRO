import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import TabbedShell from "./components/layout/TabbedShell";

export default function App(){
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/trading" replace />} />
        <Route path="/analytics" element={<Navigate to="/analytics/equity" replace />} />
        <Route path="/*" element={<TabbedShell />} />
      </Routes>
    </BrowserRouter>
  )
}
