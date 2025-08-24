import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Dashboard } from "./components/Dashboard";
import TabbedShell from "./components/layout/TabbedShell";

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#1976d2',
    },
  },
});

export default function App(){
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/trading" element={<Navigate to="/trading" replace />} />
          <Route path="/analytics" element={<Navigate to="/analytics/equity" replace />} />
          <Route path="/*" element={<TabbedShell />} />
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  )
}
