import React, { useEffect, useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Chip,
  LinearProgress,
  Alert,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  CheckCircleOutlined,
  WarningOutlined,
  CloseCircleOutlined,
  DisconnectOutlined,
  ReloadOutlined,
  StockOutlined,
  HddOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
} from '@ant-design/icons';
import { useAppDispatch, useAppSelector } from '@/store';
import { fetchModuleStatus } from '@/store/slices/systemStatusSlice';

interface ModuleNode {
  id: string;
  name: string;
  status: 'healthy' | 'warning' | 'error' | 'offline';
  position: { x: number; y: number };
  connections: string[];
  metrics?: Record<string, number>;
  lastUpdate: string;
  message?: string;
}

const ARIA_MODULES: ModuleNode[] = [
  {
    id: 'data-ingestion',
    name: 'Data Ingestion',
    status: 'healthy',
    position: { x: 50, y: 100 },
    connections: ['feature-engine'],
    lastUpdate: new Date().toISOString(),
  },
  {
    id: 'mt5-connector',
    name: 'MT5 Connector',
    status: 'healthy',
    position: { x: 50, y: 200 },
    connections: ['data-ingestion'],
    lastUpdate: new Date().toISOString(),
  },
  {
    id: 'feature-engine',
    name: 'Feature Engine',
    status: 'healthy',
    position: { x: 250, y: 100 },
    connections: ['ai-models'],
    lastUpdate: new Date().toISOString(),
  },
  {
    id: 'ai-models',
    name: 'AI Models',
    status: 'healthy',
    position: { x: 450, y: 100 },
    connections: ['smc-fusion'],
    lastUpdate: new Date().toISOString(),
  },
  {
    id: 'smc-fusion',
    name: 'SMC Fusion Core',
    status: 'healthy',
    position: { x: 650, y: 100 },
    connections: ['bias-engine'],
    lastUpdate: new Date().toISOString(),
  },
  {
    id: 'bias-engine',
    name: 'Bias Engine',
    status: 'healthy',
    position: { x: 850, y: 100 },
    connections: ['auto-trader'],
    lastUpdate: new Date().toISOString(),
  },
  {
    id: 'auto-trader',
    name: 'Auto Trader',
    status: 'healthy',
    position: { x: 1050, y: 100 },
    connections: ['trade-memory'],
    lastUpdate: new Date().toISOString(),
  },
  {
    id: 'trade-memory',
    name: 'Trade Memory',
    status: 'healthy',
    position: { x: 1250, y: 100 },
    connections: [],
    lastUpdate: new Date().toISOString(),
  },
  {
    id: 'websocket',
    name: 'WebSocket',
    status: 'healthy',
    position: { x: 450, y: 300 },
    connections: [],
    lastUpdate: new Date().toISOString(),
  },
  {
    id: 'monitoring',
    name: 'Monitoring',
    status: 'healthy',
    position: { x: 650, y: 300 },
    connections: [],
    lastUpdate: new Date().toISOString(),
  },
];

const getStatusColor = (status: string) => {
  switch (status) {
    case 'healthy': return '#00ff41';
    case 'warning': return '#ffaa00';
    case 'error': return '#ff0040';
    case 'offline': return '#666666';
    default: return '#666666';
  }
};

const getStatusIcon = (status: string) => {
  switch (status) {
    case 'healthy': return <CheckCircleOutlined style={{ color: '#00ff41' }} />;
    case 'warning': return <WarningOutlined style={{ color: '#ffaa00' }} />;
    case 'error': return <CloseCircleOutlined style={{ color: '#ff0040' }} />;
    case 'offline': return <DisconnectOutlined style={{ color: '#666666' }} />;
    default: return <DisconnectOutlined style={{ color: '#666666' }} />;
  }
};

export const AriaFlowMonitor: React.FC = () => {
  const dispatch = useAppDispatch();
  const { modules, health, loading, lastUpdate } = useAppSelector(state => state.systemStatus);
  const [localModules, setLocalModules] = useState<ModuleNode[]>(ARIA_MODULES);

  useEffect(() => {
    const fetchStatus = () => {
      dispatch(fetchModuleStatus());
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, [dispatch]);

  useEffect(() => {
    // Update local modules with real status data
    if (modules.length > 0) {
      setLocalModules(prev => prev.map(module => {
        const moduleNamePart = module.name.toLowerCase().split(' ')[0] || '';
        const realModule = modules.find(m => 
          m.name.toLowerCase().includes(moduleNamePart) ||
          module.name.toLowerCase().includes(m.name.toLowerCase())
        );
        
        if (realModule) {
          const updated: ModuleNode = {
            ...module,
            status: realModule.status,
            lastUpdate: realModule.lastUpdate,
          };
          if (realModule.message !== undefined) {
            updated.message = realModule.message;
          }
          if (realModule.metrics !== undefined) {
            updated.metrics = realModule.metrics;
          }
          return updated;
        }
        return module;
      }));
    }
  }, [modules]);

  const handleRefresh = () => {
    dispatch(fetchModuleStatus());
  };

  const healthyCount = localModules.filter(m => m.status === 'healthy').length;
  const totalCount = localModules.length;
  const healthPercentage = (healthyCount / totalCount) * 100;

  return (
    <Box sx={{ 
      p: 3, 
      backgroundColor: '#0a0a0a', 
      minHeight: '100vh',
      color: '#00ff41',
      fontFamily: 'Monaco, "Lucida Console", monospace'
    }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" sx={{ 
          color: '#00ff41', 
          textShadow: '0 0 10px #00ff41',
          fontWeight: 'bold'
        }}>
          ARIA FLOW MONITOR
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Chip 
            label={`${healthyCount}/${totalCount} OPERATIONAL`}
            sx={{ 
              backgroundColor: healthPercentage > 80 ? '#00ff41' : '#ffaa00',
              color: '#000',
              fontWeight: 'bold'
            }}
          />
          <Tooltip title="Refresh Status">
            <IconButton onClick={handleRefresh} sx={{ color: '#00ff41' }}>
              <ReloadOutlined />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* System Health Overview */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={3}>
          <Card sx={{ backgroundColor: '#1a1a1a', border: '1px solid #00ff41' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <StockOutlined style={{ color: '#00ff41' }} />
                <Typography variant="h6" sx={{ color: '#00ff41' }}>
                  SYSTEM HEALTH
                </Typography>
              </Box>
              <Typography variant="h3" sx={{ color: '#00ff41', mt: 1 }}>
                {healthPercentage.toFixed(0)}%
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={healthPercentage}
                sx={{ 
                  mt: 1,
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: healthPercentage > 80 ? '#00ff41' : '#ffaa00'
                  }
                }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card sx={{ backgroundColor: '#1a1a1a', border: '1px solid #00ff41' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <HddOutlined style={{ color: '#00ff41' }} />
                <Typography variant="h6" sx={{ color: '#00ff41' }}>
                  MEMORY USAGE
                </Typography>
              </Box>
              <Typography variant="h3" sx={{ color: '#00ff41', mt: 1 }}>
                {health?.memoryUsage?.toFixed(1) || '0.0'}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card sx={{ backgroundColor: '#1a1a1a', border: '1px solid #00ff41' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <ThunderboltOutlined style={{ color: '#00ff41' }} />
                <Typography variant="h6" sx={{ color: '#00ff41' }}>
                  CPU USAGE
                </Typography>
              </Box>
              <Typography variant="h3" sx={{ color: '#00ff41', mt: 1 }}>
                {health?.cpuUsage?.toFixed(1) || '0.0'}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card sx={{ backgroundColor: '#1a1a1a', border: '1px solid #00ff41' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <DatabaseOutlined style={{ color: '#00ff41' }} />
                <Typography variant="h6" sx={{ color: '#00ff41' }}>
                  UPTIME
                </Typography>
              </Box>
              <Typography variant="h3" sx={{ color: '#00ff41', mt: 1 }}>
                {health?.uptime ? Math.floor(health.uptime / 3600000) : '0'}h
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Module Status Grid */}
      <Typography variant="h5" sx={{ color: '#00ff41', mb: 2, textShadow: '0 0 5px #00ff41' }}>
        MODULE STATUS MATRIX
      </Typography>

      <Grid container spacing={2}>
        {localModules.map((module) => (
          <Grid item xs={12} sm={6} md={4} lg={3} key={module.id}>
            <Card sx={{ 
              backgroundColor: '#1a1a1a', 
              border: `2px solid ${getStatusColor(module.status)}`,
              boxShadow: `0 0 10px ${getStatusColor(module.status)}`,
              transition: 'all 0.3s ease'
            }}>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                  <Typography variant="h6" sx={{ 
                    color: getStatusColor(module.status),
                    fontSize: '0.9rem',
                    fontWeight: 'bold'
                  }}>
                    {module.name.toUpperCase()}
                  </Typography>
                  {getStatusIcon(module.status)}
                </Box>

                <Typography variant="body2" sx={{ color: '#888', mb: 1 }}>
                  Status: {module.status.toUpperCase()}
                </Typography>

                {module.message && (
                  <Typography variant="body2" sx={{ color: '#ccc', mb: 1 }}>
                    {module.message}
                  </Typography>
                )}

                <Typography variant="caption" sx={{ color: '#666' }}>
                  Last Update: {new Date(module.lastUpdate).toLocaleTimeString()}
                </Typography>

                {module.metrics && Object.keys(module.metrics).length > 0 && (
                  <Box sx={{ mt: 1 }}>
                    {Object.entries(module.metrics).slice(0, 2).map(([key, value]) => (
                      <Typography key={key} variant="caption" sx={{ 
                        display: 'block', 
                        color: '#00ff41',
                        fontSize: '0.7rem'
                      }}>
                        {key}: {typeof value === 'number' ? value.toFixed(2) : value}
                      </Typography>
                    ))}
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Data Flow Visualization */}
      <Box sx={{ mt: 4 }}>
        <Typography variant="h5" sx={{ color: '#00ff41', mb: 2, textShadow: '0 0 5px #00ff41' }}>
          DATA FLOW PIPELINE
        </Typography>
        
        <Card sx={{ backgroundColor: '#1a1a1a', border: '1px solid #00ff41', p: 2 }}>
          <svg width="100%" height="400" viewBox="0 0 1400 400">
            {/* Draw connections */}
            {localModules.map(module => 
              module.connections.map(connectionId => {
                const targetModule = localModules.find(m => m.id === connectionId);
                if (!targetModule) return null;
                
                return (
                  <line
                    key={`${module.id}-${connectionId}`}
                    x1={module.position.x}
                    y1={module.position.y}
                    x2={targetModule.position.x}
                    y2={targetModule.position.y}
                    stroke={getStatusColor(module.status)}
                    strokeWidth="2"
                    opacity="0.6"
                  />
                );
              })
            )}
            
            {/* Draw modules */}
            {localModules.map(module => (
              <g key={module.id}>
                <circle
                  cx={module.position.x}
                  cy={module.position.y}
                  r="20"
                  fill={getStatusColor(module.status)}
                  stroke={getStatusColor(module.status)}
                  strokeWidth="2"
                  opacity="0.8"
                />
                <text
                  x={module.position.x}
                  y={module.position.y + 35}
                  textAnchor="middle"
                  fill={getStatusColor(module.status)}
                  fontSize="10"
                  fontFamily="Monaco, monospace"
                >
                  {module.name.split(' ')[0]}
                </text>
              </g>
            ))}
          </svg>
        </Card>
      </Box>

      {loading && (
        <Alert severity="info" sx={{ mt: 2, backgroundColor: '#1a1a1a', color: '#00ff41' }}>
          Updating system status...
        </Alert>
      )}

      {lastUpdate && (
        <Typography variant="caption" sx={{ color: '#666', mt: 2, display: 'block' }}>
          Last system scan: {new Date(lastUpdate).toLocaleString()}
        </Typography>
      )}
    </Box>
  );
};
