import React, { useCallback, useEffect, useState } from "react";
import { HUD } from "../../theme/hud";
import { apiGet, apiPost } from "../../services/api";

interface SystemStatus {
  mt5_connected: boolean;
  auto_trading_enabled: boolean;
  signal_generator_active: boolean;
  risk_engine_active: boolean;
  websocket_connections: number;
  uptime_seconds: number;
  cpu_usage: number;
  memory_usage: number;
}

interface ServiceControl {
  name: string;
  status: 'running' | 'stopped' | 'error';
  description: string;
  endpoint: string;
}

const BackendControlPanel: React.FC = () => {
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [services, setServices] = useState<ServiceControl[]>([
    { name: 'MT5 Connection', status: 'stopped', description: 'MetaTrader 5 API connection', endpoint: '/api/system/mt5/toggle' },
    { name: 'Auto Trading', status: 'stopped', description: 'Automated trading execution', endpoint: '/api/institutional-ai/auto-trading/toggle' },
    { name: 'Signal Generator', status: 'stopped', description: 'AI signal generation service', endpoint: '/api/system/signals/toggle' },
    { name: 'Risk Engine', status: 'stopped', description: 'Risk management system', endpoint: '/api/system/risk/toggle' },
    { name: 'WebSocket Server', status: 'stopped', description: 'Real-time data streaming', endpoint: '/api/system/websocket/toggle' },
  ]);
  const [loading, setLoading] = useState(false);

  const fetchSystemStatus = useCallback(async () => {
    try {
      const response = await apiGet<SystemStatus>("/api/system/status");
      if (response) {
        setSystemStatus(response);
        // Update service statuses based on system status
        setServices(prev => prev.map(service => {
          let status: 'running' | 'stopped' | 'error' = 'stopped';
          switch (service.name) {
            case 'MT5 Connection':
              status = response.mt5_connected ? 'running' : 'stopped';
              break;
            case 'Auto Trading':
              status = response.auto_trading_enabled ? 'running' : 'stopped';
              break;
            case 'Signal Generator':
              status = response.signal_generator_active ? 'running' : 'stopped';
              break;
            case 'Risk Engine':
              status = response.risk_engine_active ? 'running' : 'stopped';
              break;
            case 'WebSocket Server':
              status = response.websocket_connections > 0 ? 'running' : 'stopped';
              break;
          }
          return { ...service, status };
        }));
      }
    } catch (error) {
      console.error("Failed to fetch system status:", error);
    }
  }, []);

  const toggleService = async (service: ServiceControl) => {
    setLoading(true);
    try {
      await apiPost(service.endpoint, { enabled: service.status !== 'running' });
      await fetchSystemStatus();
    } catch (error) {
      console.error(`Failed to toggle ${service.name}:`, error);
    } finally {
      setLoading(false);
    }
  };

  const restartSystem = async () => {
    if (!confirm("Restart the entire ARIA system? This will temporarily interrupt all services.")) return;
    setLoading(true);
    try {
      await apiPost("/api/system/restart", {});
      setTimeout(fetchSystemStatus, 5000); // Wait for restart
    } catch (error) {
      console.error("Failed to restart system:", error);
    } finally {
      setLoading(false);
    }
  };

  const emergencyStop = async () => {
    if (!confirm("EMERGENCY STOP: This will immediately halt all trading and AI services. Continue?")) return;
    setLoading(true);
    try {
      await apiPost("/api/system/emergency-stop", {});
      await fetchSystemStatus();
    } catch (error) {
      console.error("Failed to execute emergency stop:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSystemStatus();
    const interval = setInterval(fetchSystemStatus, 10000);
    return () => clearInterval(interval);
  }, [fetchSystemStatus]);

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'text-emerald-300 border-emerald-400/40';
      case 'error': return 'text-rose-300 border-rose-400/40';
      default: return 'text-amber-300 border-amber-400/40';
    }
  };

  return (
    <div className={`${HUD.CARD} rounded-xl p-4`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-cyan-200 font-semibold">Backend Control Panel</h3>
        <div className="flex gap-2">
          <button
            onClick={fetchSystemStatus}
            disabled={loading}
            className={`${HUD.TAB} border border-cyan-400/20 text-xs`}
          >
            {loading ? "Refreshing..." : "Refresh"}
          </button>
          <button
            onClick={restartSystem}
            disabled={loading}
            className="px-3 py-1 text-xs bg-amber-600/70 hover:bg-amber-600 rounded border border-amber-400/30"
          >
            Restart System
          </button>
          <button
            onClick={emergencyStop}
            disabled={loading}
            className="px-3 py-1 text-xs bg-rose-600/70 hover:bg-rose-600 rounded border border-rose-400/30"
          >
            Emergency Stop
          </button>
        </div>
      </div>

      {/* System Overview */}
      {systemStatus && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          <div className={`${HUD.CARD} p-3 rounded-lg`}>
            <div className={HUD.TITLE}>Uptime</div>
            <div className={HUD.VALUE}>{formatUptime(systemStatus.uptime_seconds)}</div>
          </div>
          <div className={`${HUD.CARD} p-3 rounded-lg`}>
            <div className={HUD.TITLE}>CPU Usage</div>
            <div className={`${HUD.VALUE} ${systemStatus.cpu_usage > 80 ? 'text-rose-300' : 'text-emerald-300'}`}>
              {systemStatus.cpu_usage.toFixed(1)}%
            </div>
          </div>
          <div className={`${HUD.CARD} p-3 rounded-lg`}>
            <div className={HUD.TITLE}>Memory Usage</div>
            <div className={`${HUD.VALUE} ${systemStatus.memory_usage > 80 ? 'text-rose-300' : 'text-emerald-300'}`}>
              {systemStatus.memory_usage.toFixed(1)}%
            </div>
          </div>
          <div className={`${HUD.CARD} p-3 rounded-lg`}>
            <div className={HUD.TITLE}>WS Connections</div>
            <div className={HUD.VALUE}>{systemStatus.websocket_connections}</div>
          </div>
        </div>
      )}

      {/* Service Controls */}
      <div className="space-y-2">
        <h4 className="text-cyan-300 font-medium mb-2">Service Management</h4>
        {services.map((service) => (
          <div key={service.name} className={`${HUD.CARD} p-3 rounded-lg flex items-center justify-between`}>
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${service.status === 'running' ? 'bg-emerald-400' : service.status === 'error' ? 'bg-rose-400' : 'bg-amber-400'}`} />
                <span className="text-cyan-100 font-medium">{service.name}</span>
                <span className={`text-xs px-2 py-1 rounded border ${getStatusColor(service.status)}`}>
                  {service.status.toUpperCase()}
                </span>
              </div>
              <div className="text-xs text-cyan-300/70 mt-1">{service.description}</div>
            </div>
            <button
              onClick={() => toggleService(service)}
              disabled={loading}
              className={`px-3 py-1 text-xs rounded border ${
                service.status === 'running' 
                  ? 'bg-rose-600/70 hover:bg-rose-600 border-rose-400/30' 
                  : 'bg-emerald-600/70 hover:bg-emerald-600 border-emerald-400/30'
              }`}
            >
              {service.status === 'running' ? 'Stop' : 'Start'}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default BackendControlPanel;
