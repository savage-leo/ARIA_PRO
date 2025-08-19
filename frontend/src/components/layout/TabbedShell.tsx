// frontend/src/components/layout/TabbedShell.tsx
import React, { useEffect, useState } from "react";
import { Tab } from "@headlessui/react";
import { useLocation, useNavigate } from "react-router-dom";
import TradingInterface from "../interfaces/TradingInterface";
import InstitutionalAITab from "../interfaces/InstitutionalAITab";
import FlowMonitorTab from "../interfaces/FlowMonitorTab";
import OrdersTab from "../interfaces/OrdersTab";
import PositionsTab from "../interfaces/PositionsTab";
import WatchlistTab from "../interfaces/WatchlistTab";
import SettingsTab from "../interfaces/SettingsTab";
import { HUD } from "../../theme/hud";

type TabDef = { route: string; label: string; element: React.ReactNode };

const tabs: TabDef[] = [
  { route: "/trading",                 label: "Trading Interface",       element: <TradingInterface /> },
  { route: "/flow-monitor",            label: "ARIA Flow Monitor",      element: <FlowMonitorTab /> },
  { route: "/orders",                  label: "Order Flow & Sentiment", element: <OrdersTab /> },
  { route: "/positions",               label: "Positions & Equity",     element: <PositionsTab /> },
  { route: "/watchlist",               label: "Market Watchlist",       element: <WatchlistTab /> },
  { route: "/ai-smc",                  label: "Institutional AI",      element: <InstitutionalAITab /> },
  { route: "/settings",                label: "System Settings",        element: <SettingsTab /> },
];

const TabbedShell: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const initialIndex = Math.max(0, tabs.findIndex((t) => location.pathname.startsWith(t.route)));
  const [selectedIndex, setSelectedIndex] = useState(initialIndex);

  useEffect(() => {
    const idx = Math.max(0, tabs.findIndex((t) => location.pathname.startsWith(t.route)));
    setSelectedIndex(idx);
  }, [location.pathname]);

  return (
    <div className={`h-full flex flex-col ${HUD.BG}`}>
      <Tab.Group selectedIndex={selectedIndex} onChange={(i) => {
        setSelectedIndex(i);
        const route = tabs[i]?.route || "/trading";
        if (location.pathname !== route) navigate(route, { replace: true });
      }}>
        <Tab.List className="flex gap-2 p-2 border-b border-cyan-400/10 bg-slate-900/40">
          {tabs.map((t) => (
            <Tab
              key={t.route}
              className={({ selected }) => (selected ? HUD.TAB_ACTIVE : HUD.TAB)}
            >
              {t.label}
            </Tab>
          ))}
        </Tab.List>
        <Tab.Panels className="flex-1 overflow-auto">
          {tabs.map((t) => (
            <Tab.Panel key={t.route} className="p-2">{t.element}</Tab.Panel>
          ))}
        </Tab.Panels>
      </Tab.Group>
    </div>
  );
};

export default TabbedShell;
