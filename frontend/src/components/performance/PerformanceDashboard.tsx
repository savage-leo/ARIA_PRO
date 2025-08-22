import React, { useEffect, useState } from 'react';
import { Card, Row, Col, Statistic, Tabs, Alert, Progress, Tag } from 'antd';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import type { ChartOptions, ChartData } from 'chart.js';
import {
  DashboardOutlined,
  ClockCircleOutlined,
  AlertOutlined,
  ThunderboltOutlined,
  DeploymentUnitOutlined,
  FundOutlined,
} from '@ant-design/icons';
import { useWebSocket } from '../../hooks/useWebSocket';
import { backendBase } from '../../services/api';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

// Tabs uses items API (AntD v5)

interface ModelMetrics {
  model: string;
  calls: number;
  latency_ms: {
    min: number;
    max: number;
    avg: number;
    ema: number;
  };
  memory_mb: {
    current: number;
    avg: number;
    p95: number;
  };
  cpu_percent: {
    current: number;
    avg: number;
    p95: number;
  };
  errors: number;
  last_call: string;
}

interface SystemMetrics {
  system: {
    uptime_seconds: number;
    cpu_percent: number;
    memory: {
      rss_mb: number;
      vms_mb: number;
      available_percent: number;
    };
    threads: number;
    open_files: number;
  };
  models_tracked: number;
  total_inferences: number;
  symbols_tracked?: number;
  total_symbol_inferences?: number;
}

interface PerfAlert {
  type: string;
  message: string;
  timestamp: string;
  severity: string;
}

interface Thresholds {
  cpu_warn_percent: number;
  mem_available_percent_crit: number;
  latency_warn_ms: number;
}

type PerfWsMessage =
  | { type: 'model_metrics'; model: string; metrics: ModelMetrics }
  | { type: 'system_metrics'; metrics: SystemMetrics }
  | { type: 'symbol_metrics'; symbol: string; model: string; metrics: ModelMetrics }
  | { type: 'alert'; message: string; severity: string; timestamp: string };

export const PerformanceDashboard: React.FC = () => {
  const [modelMetrics, setModelMetrics] = useState<Record<string, ModelMetrics>>({});
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [alerts, setAlerts] = useState<PerfAlert[]>([]);
  const [latencyHistory, setLatencyHistory] = useState<Record<string, number[]>>({});
  const [cpuHistory, setCpuHistory] = useState<Record<string, number[]>>({});
  const [symbolMetrics, setSymbolMetrics] = useState<Record<string, Record<string, ModelMetrics>>>({});
  const [symbolLatencyHistory, setSymbolLatencyHistory] = useState<Record<string, Record<string, number[]>>>({});
  const [symbolCpuHistory, setSymbolCpuHistory] = useState<Record<string, Record<string, number[]>>>({});
  const [thresholds, setThresholds] = useState<Thresholds | null>(null);

  // WebSocket connection for real-time updates
  const proto = (typeof window !== 'undefined' && window.location.protocol === 'https:') ? 'wss' : 'ws';
  const host = (typeof window !== 'undefined') ? window.location.host : 'localhost:5175';
  const wsUrl = (backendBase && /^https?:\/\//.test(backendBase))
    ? backendBase.replace(/^http(s?):\/\//, (_m, s) => (s ? 'wss://' : 'ws://')).replace(/\/+$/, '') + '/monitoring/ws/performance'
    : `${proto}://${host}/monitoring/ws/performance`;
  const { lastMessage, isConnected, isConnecting, error } = useWebSocket({ url: wsUrl });

  useEffect(() => {
    if (!lastMessage) return;
    try {
      const data = lastMessage as unknown as PerfWsMessage;
      switch (data.type) {
        case 'model_metrics':
          setModelMetrics(prev => ({
            ...prev,
            [data.model]: data.metrics
          }));

          // Update history for charts
          setLatencyHistory(prev => ({
            ...prev,
            [data.model]: [...(prev[data.model] || []), data.metrics.latency_ms.ema].slice(-50)
          }));

          setCpuHistory(prev => ({
            ...prev,
            [data.model]: [...(prev[data.model] || []), data.metrics.cpu_percent.current].slice(-50)
          }));
          break;

        case 'system_metrics':
          setSystemMetrics(data.metrics);
          break;

        case 'symbol_metrics': {
          const sym: string = data.symbol;
          const mdl: string = data.model;
          const metrics: ModelMetrics = data.metrics;
          setSymbolMetrics(prev => ({
            ...prev,
            [sym]: {
              ...(prev[sym] || {}),
              [mdl]: metrics,
            },
          }));

          // histories per symbol/model
          setSymbolLatencyHistory(prev => {
            const symMap = prev[sym] || {};
            const arr = [...(symMap[mdl] || []), metrics.latency_ms.ema].slice(-50);
            return { ...prev, [sym]: { ...symMap, [mdl]: arr } };
          });
          setSymbolCpuHistory(prev => {
            const symMap = prev[sym] || {};
            const arr = [...(symMap[mdl] || []), metrics.cpu_percent.current].slice(-50);
            return { ...prev, [sym]: { ...symMap, [mdl]: arr } };
          });
          break;
        }

        case 'alert': {
          const alertItem: PerfAlert = {
            type: 'alert',
            message: data.message,
            severity: data.severity,
            timestamp: data.timestamp,
          };
          setAlerts(prev => [alertItem, ...prev].slice(0, 10));
          break;
        }
      }
    } catch (err) {
      console.error('Error handling WebSocket message:', err);
    }
  }, [lastMessage]);

  // Bootstrap initial metrics and thresholds from REST
  useEffect(() => {
    const loadInitial = async () => {
      try {
        const res = await fetch(`${backendBase}/monitoring/performance/metrics`);
        if (!res.ok) return;
        const payload = await res.json();
        if (payload.system) setSystemMetrics(payload.system);
        if (payload.models) setModelMetrics(payload.models);
        if (payload.symbols) setSymbolMetrics(payload.symbols);
        if (payload.thresholds) setThresholds(payload.thresholds as Thresholds);
      } catch (e) {
        console.warn('Failed to bootstrap metrics:', e);
      }
    };
    loadInitial();
  }, []);

  // Generate chart data for a specific model
  const getLatencyChartData = (model: string): ChartData<'line'> => ({
    labels: Array.from({ length: latencyHistory[model]?.length || 0 }, (_, i) => i),
    datasets: [{
      label: `${model} Latency (ms)`,
      data: latencyHistory[model] || [],
      borderColor: '#1890ff',
      backgroundColor: 'rgba(24, 144, 255, 0.1)',
      tension: 0.4,
      fill: true,
    }]
  });

  const getCpuChartData = (model: string): ChartData<'line'> => ({
    labels: Array.from({ length: cpuHistory[model]?.length || 0 }, (_, i) => i),
    datasets: [{
      label: `${model} CPU (%)`,
      data: cpuHistory[model] || [],
      borderColor: '#52c41a',
      backgroundColor: 'rgba(82, 196, 26, 0.1)',
      tension: 0.4,
      fill: true,
    }]
  });

  // Symbol-level charts
  const getSymbolLatencyChartData = (symbol: string, model: string): ChartData<'line'> => ({
    labels: Array.from({ length: symbolLatencyHistory[symbol]?.[model]?.length || 0 }, (_, i) => i),
    datasets: [{
      label: `${symbol}/${model} Latency (ms)`,
      data: (symbolLatencyHistory[symbol] && symbolLatencyHistory[symbol][model]) ? symbolLatencyHistory[symbol][model] : [],
      borderColor: '#fa8c16',
      backgroundColor: 'rgba(250, 140, 22, 0.1)',
      tension: 0.4,
      fill: true,
    }]
  });

  const getSymbolCpuChartData = (symbol: string, model: string): ChartData<'line'> => ({
    labels: Array.from({ length: symbolCpuHistory[symbol]?.[model]?.length || 0 }, (_, i) => i),
    datasets: [{
      label: `${symbol}/${model} CPU (%)`,
      data: (symbolCpuHistory[symbol] && symbolCpuHistory[symbol][model]) ? symbolCpuHistory[symbol][model] : [],
      borderColor: '#722ed1',
      backgroundColor: 'rgba(114, 46, 209, 0.1)',
      tension: 0.4,
      fill: true,
    }]
  });

  const chartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
    },
    scales: {
      x: {
        display: false,
      },
      y: {
        beginAtZero: true,
      },
    },
  };

  const getStatusColor = (value: number, thresholds: { warning: number; critical: number }) => {
    // Map to valid antd Progress statuses
    if (value >= thresholds.warning) return 'exception';
    return 'success';
  };

  return (
    <div className="performance-dashboard" style={{ padding: '24px' }}>
      <Row gutter={[16, 16]}>
        {/* Connection Status */}
        <Col span={24}>
          <Alert
            message={`WebSocket: ${isConnected ? 'Connected' : (isConnecting ? 'Connecting' : (error ? `Error: ${error}` : 'Disconnected'))}`}
            type={isConnected ? 'success' : (error ? 'error' : 'warning')}
            showIcon
            style={{ marginBottom: 16 }}
          />
        </Col>

        {/* System Overview */}
        <Col span={24}>
          <Card title={<><DashboardOutlined /> System Overview</>} bordered={false}>
            <Row gutter={16}>
              <Col span={4}>
                <Statistic 
                  title="CPU Usage" 
                  value={systemMetrics?.system.cpu_percent ?? 0} 
                  suffix="%" 
                  valueStyle={{ 
                    color: ((systemMetrics?.system.cpu_percent ?? 0) > (thresholds?.cpu_warn_percent ?? 80)) ? '#cf1322' : '#3f8600' 
                  }}
                />
                <Progress 
                  percent={systemMetrics?.system.cpu_percent ?? 0} 
                  size="small" 
                  status={getStatusColor(systemMetrics?.system.cpu_percent ?? 0, { warning: 70, critical: 90 })}
                  showInfo={false}
                />
              </Col>
              <Col span={4}>
                <Statistic 
                  title="Memory Usage" 
                  value={100 - (systemMetrics?.system.memory.available_percent || 100)} 
                  suffix="%" 
                  valueStyle={{ 
                    color: (100 - (systemMetrics?.system.memory.available_percent || 100)) > (100 - (thresholds?.mem_available_percent_crit ?? 15)) ? '#cf1322' : '#3f8600' 
                  }}
                />
                <Progress 
                  percent={100 - (systemMetrics?.system.memory.available_percent || 100)} 
                  size="small" 
                  status={getStatusColor(100 - (systemMetrics?.system.memory.available_percent || 100), { warning: 80, critical: 90 })}
                  showInfo={false}
                />
              </Col>
              <Col span={4}>
                <Statistic 
                  title="Active Threads" 
                  value={systemMetrics?.system.threads || 0} 
                  prefix={<ThunderboltOutlined />}
                />
              </Col>
              <Col span={4}>
                <Statistic
                  title="Models Tracked" 
                  value={systemMetrics?.models_tracked || 0} 
                  prefix={<DeploymentUnitOutlined />}
                />
              </Col>
              <Col span={4}>
                <Statistic
                  title="Total Inferences" 
                  value={systemMetrics?.total_inferences || 0} 
                  prefix={<FundOutlined />}
                />
              </Col>
              <Col span={4}>
                <Statistic 
                  title="Uptime" 
                  value={Math.floor((systemMetrics?.system.uptime_seconds || 0) / 3600)} 
                  suffix="h"
                  prefix={<ClockCircleOutlined />}
                />
              </Col>
            </Row>
            {(systemMetrics?.symbols_tracked !== undefined || systemMetrics?.total_symbol_inferences !== undefined) && (
              <Row style={{ marginTop: 16 }}>
                <Col span={24}>
                  <Tag color="blue" style={{ marginRight: 8 }}>
                    Symbols: {systemMetrics?.symbols_tracked ?? 0}
                  </Tag>
                  <Tag color="purple">
                    Symbol Inferences: {systemMetrics?.total_symbol_inferences ?? 0}
                  </Tag>
                </Col>
              </Row>
            )}
          </Card>
        </Col>

        {/* Alerts */}
        {alerts.length > 0 && (
          <Col span={24}>
            <Card title={<><AlertOutlined /> Performance Alerts</>} bordered={false}>
              {alerts.slice(0, 5).map((alert, i) => (
                <Alert
                  key={i}
                  message={alert.message}
                  type={alert.severity === 'critical' ? 'error' : 'warning'}
                  showIcon
                  style={{ marginBottom: 8 }}
                  description={`${new Date(alert.timestamp).toLocaleTimeString()}`}
                />
              ))}
            </Card>
          </Col>
        )}

        {/* Model Performance */}
        <Col span={24}>
          <Card title="Model Performance Metrics" bordered={false}>
            <Tabs
              defaultActiveKey="overview"
              items={[
                {
                  key: 'overview',
                  label: 'Overview',
                  children: (
                    <Row gutter={[16, 16]}>
                      {Object.entries(modelMetrics).map(([model, metrics]) => (
                        <Col span={8} key={model}>
                          <Card size="small" title={model} bordered>
                            <Row gutter={[8, 8]}>
                              <Col span={12}>
                                <Statistic title="Calls" value={metrics.calls} valueStyle={{ fontSize: '14px' }} />
                              </Col>
                              <Col span={12}>
                                <Statistic
                                  title="Errors"
                                  value={metrics.errors}
                                  valueStyle={{ fontSize: '14px', color: metrics.errors > 0 ? '#cf1322' : '#3f8600' }}
                                />
                              </Col>
                              <Col span={12}>
                                <Statistic title="Avg Latency" value={metrics.latency_ms.avg.toFixed(1)} suffix="ms" valueStyle={{ fontSize: '14px' }} />
                              </Col>
                              <Col span={12}>
                                <Statistic
                                  title="EMA Latency"
                                  value={metrics.latency_ms.ema.toFixed(1)}
                                  suffix="ms"
                                  valueStyle={{ fontSize: '14px', color: metrics.latency_ms.ema > (thresholds?.latency_warn_ms ?? 1000) ? '#cf1322' : '#3f8600' }}
                                />
                              </Col>
                              <Col span={12}>
                                <Statistic title="CPU P95" value={metrics.cpu_percent.p95.toFixed(1)} suffix="%" valueStyle={{ fontSize: '14px' }} />
                              </Col>
                              <Col span={12}>
                                <Statistic title="Memory P95" value={metrics.memory_mb.p95.toFixed(1)} suffix="MB" valueStyle={{ fontSize: '14px' }} />
                              </Col>
                            </Row>
                          </Card>
                        </Col>
                      ))}
                    </Row>
                  )
                },
                ...Object.keys(modelMetrics).map((model) => ({
                  key: model,
                  label: model,
                  children: (
                    <Row gutter={[16, 16]}>
                      <Col span={12}>
                        <Card title="Latency Trend" size="small">
                          <div style={{ height: 200 }}>
                            <Line data={getLatencyChartData(model)} options={chartOptions} />
                          </div>
                        </Card>
                      </Col>
                      <Col span={12}>
                        <Card title="CPU Usage Trend" size="small">
                          <div style={{ height: 200 }}>
                            <Line data={getCpuChartData(model)} options={chartOptions} />
                          </div>
                        </Card>
                      </Col>
                      <Col span={24}>
                        <Card title="Detailed Metrics" size="small">
                          <Row gutter={16}>
                            <Col span={6}><Statistic title="Total Calls" value={modelMetrics[model]?.calls ?? 0} /></Col>
                            <Col span={6}><Statistic title="Min Latency" value={(modelMetrics[model]?.latency_ms?.min ?? 0).toFixed(2)} suffix="ms" /></Col>
                            <Col span={6}><Statistic title="Max Latency" value={(modelMetrics[model]?.latency_ms?.max ?? 0).toFixed(2)} suffix="ms" /></Col>
                            <Col span={6}><Statistic title="Last Call" value={modelMetrics[model]?.last_call ? new Date(modelMetrics[model].last_call).toLocaleTimeString() : 'N/A'} /></Col>
                          </Row>
                        </Card>
                      </Col>
                    </Row>
                  )
                }))
              ]}
            />
          </Card>
        </Col>
        
        {/* Per-Symbol Performance */}
        <Col span={24}>
          <Card title="Per-Symbol Performance Metrics" bordered={false}>
            <Tabs
              defaultActiveKey="symbols-overview"
              items={[
                {
                  key: 'symbols-overview',
                  label: 'Overview',
                  children: (
                    <Row gutter={[16, 16]}>
                      {Object.entries(symbolMetrics).map(([symbol, models]) => (
                        <Col span={8} key={symbol}>
                          <Card size="small" title={symbol} bordered>
                            {Object.entries(models).map(([model, metrics]) => (
                              <div key={`${symbol}-${model}`} style={{ marginBottom: 12 }}>
                                <Row gutter={[8, 8]}>
                                  <Col span={12}>
                                    <Statistic title={`${model} Calls`} value={metrics.calls} valueStyle={{ fontSize: '13px' }} />
                                  </Col>
                                  <Col span={12}>
                                    <Statistic
                                      title="EMA Latency"
                                      value={metrics.latency_ms.ema.toFixed(1)}
                                      suffix="ms"
                                      valueStyle={{ fontSize: '13px', color: metrics.latency_ms.ema > (thresholds?.latency_warn_ms ?? 1000) ? '#cf1322' : '#3f8600' }}
                                    />
                                  </Col>
                                  <Col span={24}>
                                    <div style={{ height: 120 }}>
                                      <Line data={getSymbolLatencyChartData(symbol, model)} options={chartOptions} />
                                    </div>
                                  </Col>
                                </Row>
                              </div>
                            ))}
                          </Card>
                        </Col>
                      ))}
                    </Row>
                  )
                },
                ...Object.keys(symbolMetrics).map((symbol) => ({
                  key: `sym-${symbol}`,
                  label: symbol,
                  children: (
                    <Row gutter={[16, 16]}>
                      {Object.keys(symbolMetrics[symbol] || {}).map((model) => (
                        <Col span={12} key={`${symbol}-${model}`}>
                          <Card title={`${symbol} • ${model}`} size="small">
                            <Row gutter={[16, 16]}>
                              <Col span={12}>
                                <div style={{ height: 180 }}>
                                  <Line data={getSymbolLatencyChartData(symbol, model)} options={chartOptions} />
                                </div>
                              </Col>
                              <Col span={12}>
                                <div style={{ height: 180 }}>
                                  <Line data={getSymbolCpuChartData(symbol, model)} options={chartOptions} />
                                </div>
                              </Col>
                              <Col span={24}>
                                <Row gutter={16}>
                                  <Col span={8}>
                                    <Statistic title="Calls" value={symbolMetrics[symbol]?.[model]?.calls ?? 0} />
                                  </Col>
                                  <Col span={8}>
                                    <Statistic title="EMA (ms)" value={(symbolMetrics[symbol]?.[model]?.latency_ms?.ema ?? 0).toFixed(1)} />
                                  </Col>
                                  <Col span={8}>
                                    <Statistic title="CPU P95 (%)" value={(symbolMetrics[symbol]?.[model]?.cpu_percent?.p95 ?? 0).toFixed(1)} />
                                  </Col>
                                </Row>
                              </Col>
                            </Row>
                          </Card>
                        </Col>
                      ))}
                    </Row>
                  )
                }))
              ]}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};
