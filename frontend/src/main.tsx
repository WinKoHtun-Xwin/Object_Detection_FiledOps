import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import App from './App';
import { LivePage } from './pages/LivePage';
import { PeoplePage } from './pages/PeoplePage';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<App />}>
          <Route index element={<LivePage />} />
          <Route path="people" element={<PeoplePage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);
