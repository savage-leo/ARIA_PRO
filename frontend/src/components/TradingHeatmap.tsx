import React, { useEffect, useState } from 'react';
import { Box, Card, CardContent, Typography, CircularProgress, FormControl, InputLabel, Select, MenuItem } from '@mui/material';
import type { SelectChangeEvent } from '@mui/material/Select';

interface HeatmapData {
  symbol: string;
  hour: number;
  day_of_week: number;
  avg_pnl: number;
  trade_count: number;
  win_rate: number;
}

const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const HOURS = Array.from({ length: 24 }, (_, i) => i);

export const TradingHeatmap: React.FC = () => {
  const [data, setData] = useState<HeatmapData[]>([]);
  const [loading, setLoading] = useState(true);
  const [days, setDays] = useState<number>(30);
  const [error, setError] = useState<string | null>(null);

  // Environment-aware base HTTP (dev uses Vite proxy via relative paths)
  const baseHttpEnv = (import.meta.env.VITE_BACKEND_BASE as string | undefined)?.replace(/\/$/, "");
  const baseHttp = baseHttpEnv || "";

  useEffect(() => {
    const fetchHeatmapData = async () => {
      setError(null);
      setLoading(true);
      try {
        const response = await fetch(`${baseHttp}/api/analytics/trading-heatmap?days=${days}`);
        const result = await response.json();
        
        if (result.ok && result.data) {
          setData(result.data);
        } else if (!result.ok) {
          setError(result.message || 'Failed to load heatmap');
        }
      } catch (error) {
        console.error('Failed to fetch heatmap data:', error);
        setError('Failed to fetch heatmap data');
      } finally {
        setLoading(false);
      }
    };

    fetchHeatmapData();
  }, [days, baseHttp]);

  const getHeatmapValue = (hour: number, dayOfWeek: number) => {
    const cellData = data.filter(d => d.hour === hour && d.day_of_week === dayOfWeek);
    if (cellData.length === 0) return null;
    
    const avgPnl = cellData.reduce((sum, d) => sum + d.avg_pnl, 0) / cellData.length;
    const totalTrades = cellData.reduce((sum, d) => sum + d.trade_count, 0);
    
    return { avgPnl, totalTrades };
  };

  const getHeatmapColor = (avgPnl: number) => {
    if (avgPnl > 0) {
      const intensity = Math.min(avgPnl / 50, 1); // Normalize to max 50 profit
      return `rgba(34, 211, 238, ${0.2 + intensity * 0.8})`; // cyan for profit
    } else {
      const intensity = Math.min(Math.abs(avgPnl) / 50, 1);
      return `rgba(255, 64, 64, ${0.2 + intensity * 0.8})`; // red for loss
    }
  };

  return (
    <Card sx={{ backgroundColor: '#0b1020', border: '1px solid rgba(34,211,238,0.3)' }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2, gap: 2, flexWrap: 'wrap' }}>
          <Typography variant="h6" sx={{ color: '#22d3ee' }}>
            TRADING PERFORMANCE HEATMAP
          </Typography>
          <FormControl size="small" sx={{ minWidth: 140 }}>
            <InputLabel sx={{ color: '#22d3ee' }}>Period</InputLabel>
            <Select
              value={days.toString()}
              onChange={(e: SelectChangeEvent) => setDays(Number(e.target.value))}
              sx={{ color: '#e0fbff', '& .MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(34,211,238,0.3)' } }}
            >
              <MenuItem value={"7"}>7 Days</MenuItem>
              <MenuItem value={"30"}>30 Days</MenuItem>
              <MenuItem value={"90"}>90 Days</MenuItem>
              <MenuItem value={"365"}>1 Year</MenuItem>
            </Select>
          </FormControl>
        </Box>

        {error && (
          <Box sx={{ color: '#ff6b6b', border: '1px solid rgba(255,107,107,0.5)', p: 1, borderRadius: 1, mb: 2 }}>
            {error}
          </Box>
        )}
        
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
            <CircularProgress sx={{ color: '#22d3ee' }} />
          </Box>
        ) : (
          <Box sx={{ overflowX: 'auto' }}>
            <Box sx={{ minWidth: 800, position: 'relative' }}>
              {/* Hour labels */}
              <Box sx={{ display: 'flex', mb: 1, ml: 4 }}>
                {HOURS.map(hour => (
                  <Box
                    key={hour}
                    sx={{
                      width: 30,
                      height: 20,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '0.7rem',
                      color: '#22d3ee',
                    }}
                  >
                    {hour}
                  </Box>
                ))}
              </Box>
              
              {/* Heatmap grid */}
              {DAYS.map((day, dayIndex) => (
                <Box key={day} sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                  {/* Day label */}
                  <Box
                    sx={{
                      width: 30,
                      height: 25,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '0.8rem',
                      color: '#22d3ee',
                      fontWeight: 'bold',
                    }}
                  >
                    {day}
                  </Box>
                  
                  {/* Hour cells */}
                  {HOURS.map(hour => {
                    const cellValue = getHeatmapValue(hour, dayIndex);
                    
                    return (
                      <Box
                        key={`${day}-${hour}`}
                        sx={{
                          width: 30,
                          height: 25,
                          backgroundColor: cellValue 
                            ? getHeatmapColor(cellValue.avgPnl)
                            : 'rgba(31, 41, 55, 0.6)',
                          border: '1px solid #1f2937',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: '0.6rem',
                          color: '#e0fbff',
                          cursor: 'pointer',
                          transition: 'all 0.2s ease',
                          '&:hover': {
                            transform: 'scale(1.1)',
                            zIndex: 10,
                            border: '1px solid rgba(34,211,238,0.6)',
                          },
                        }}
                        title={
                          cellValue
                            ? `${day} ${hour}:00\nAvg P&L: $${cellValue.avgPnl.toFixed(2)}\nTrades: ${cellValue.totalTrades}`
                            : `${day} ${hour}:00\nNo trades`
                        }
                      >
                        {cellValue && cellValue.totalTrades > 0 && (
                          <span>{cellValue.totalTrades}</span>
                        )}
                      </Box>
                    );
                  })}
                </Box>
              ))}
              
              {/* Legend */}
              <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
                <Typography variant="caption" sx={{ color: '#22d3ee' }}>
                  Legend:
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Box sx={{ width: 15, height: 15, backgroundColor: 'rgba(255, 64, 64, 0.8)' }} />
                  <Typography variant="caption" sx={{ color: '#e0fbff' }}>Loss</Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Box sx={{ width: 15, height: 15, backgroundColor: 'rgba(34, 211, 238, 0.8)' }} />
                  <Typography variant="caption" sx={{ color: '#e0fbff' }}>Profit</Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Box sx={{ width: 15, height: 15, backgroundColor: 'rgba(31, 41, 55, 0.6)' }} />
                  <Typography variant="caption" sx={{ color: '#e0fbff' }}>No Data</Typography>
                </Box>
              </Box>
            </Box>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};
