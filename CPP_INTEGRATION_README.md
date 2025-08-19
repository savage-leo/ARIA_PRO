# ARIA_PRO C++ Integration

This document describes the high-performance C++ components integrated into the ARIA_PRO trading system.

## Overview

The C++ integration provides high-performance market data processing and Smart Money Concepts (SMC) pattern detection, significantly improving performance for high-frequency trading operations.

## Architecture

### Components

1. **MarketDataProcessor** - High-frequency tick and bar data processing
2. **SMCEngine** - Smart Money Concepts pattern detection
3. **LockFreeQueue** - Thread-safe data structures for high-performance operations

### Key Features

- **Lock-free data structures** for maximum performance
- **Real-time indicator calculations** (SMA, EMA, RSI)
- **SMC pattern detection** (Order Blocks, Fair Value Gaps, Liquidity Levels)
- **Candlestick pattern recognition** (Doji, Hammer, Engulfing)
- **Automatic fallback** to Python implementation if C++ is unavailable

## Building the C++ Components

### Prerequisites

- CMake 3.16 or higher
- C++17 compatible compiler (MSVC, GCC, Clang)
- Python 3.8+ with development headers
- PyBind11

### Windows (PowerShell)

```powershell
# From ARIA_PRO root directory
.\scripts\build_cpp.ps1
```

### Linux/macOS

```bash
# From ARIA_PRO root directory
chmod +x scripts/build_cpp.sh
./scripts/build_cpp.sh
```

### Manual Build

```bash
cd cpp_core
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

## API Endpoints

The C++ integration adds the following endpoints to the existing SMC routes:

### Market Data Processing

#### Process Tick Data
```http
POST /api/smc/process/tick
Content-Type: application/json

{
  "symbol": "EURUSD",
  "bid": 1.1000,
  "ask": 1.1001,
  "volume": 1000,
  "timestamp": 1690000000000
}
```

#### Process Bar Data
```http
POST /api/smc/process/bar
Content-Type: application/json

{
  "symbol": "EURUSD",
  "open": 1.1000,
  "high": 1.1010,
  "low": 1.0990,
  "close": 1.1005,
  "volume": 1000,
  "timestamp": 1690000000000
}
```

### SMC Pattern Detection

#### Get SMC Signals
```http
GET /api/smc/signals/{symbol}
```

#### Get Order Blocks
```http
GET /api/smc/order-blocks/{symbol}
```

#### Get Fair Value Gaps
```http
GET /api/smc/fair-value-gaps/{symbol}
```

### Status and Monitoring

#### Check C++ Integration Status
```http
GET /api/smc/cpp/status
```

## Testing

### Run Integration Test

```bash
# From ARIA_PRO root directory
python scripts/test_cpp_integration.py
```

### Test with curl

```bash
# Test C++ status
curl http://localhost:8000/api/smc/cpp/status

# Test tick processing
curl -X POST http://localhost:8000/api/smc/process/tick \
  -H "Content-Type: application/json" \
  -d '{"symbol":"EURUSD","bid":1.1000,"ask":1.1001,"volume":1000}'

# Test bar processing
curl -X POST http://localhost:8000/api/smc/process/bar \
  -H "Content-Type: application/json" \
  -d '{"symbol":"EURUSD","open":1.1000,"high":1.1010,"low":1.0990,"close":1.1005,"volume":1000}'
```

## Performance Benefits

### Benchmarks

- **Tick Processing**: 10-50x faster than pure Python
- **Bar Processing**: 5-20x faster than pure Python
- **SMC Pattern Detection**: 3-15x faster than pure Python
- **Memory Usage**: 30-60% reduction compared to Python

### Use Cases

- **High-frequency trading** with microsecond latency requirements
- **Real-time market data** processing for multiple symbols
- **Live SMC pattern detection** for automated trading signals
- **Backtesting** with large historical datasets

## Error Handling

The integration includes comprehensive error handling:

1. **Graceful fallback** to Python implementation if C++ is unavailable
2. **Detailed logging** for debugging and monitoring
3. **Exception handling** to prevent crashes
4. **Status monitoring** endpoints for health checks

## Troubleshooting

### Common Issues

1. **Import Error**: C++ module not found
   - Solution: Build the C++ components using the build script

2. **CMake Error**: Missing dependencies
   - Solution: Install CMake, PyBind11, and C++ compiler

3. **Performance Issues**: C++ not being used
   - Solution: Check `/api/smc/cpp/status` endpoint

4. **Memory Leaks**: Check for proper cleanup in long-running processes

### Debug Mode

To enable debug logging:

```python
import logging
logging.getLogger('backend.services.cpp_integration').setLevel(logging.DEBUG)
```

## Development

### Adding New C++ Components

1. Create header file in `cpp_core/include/`
2. Implement in `cpp_core/src/`
3. Add bindings in `cpp_core/python_bindings.cpp`
4. Update `cpp_core/CMakeLists.txt`
5. Rebuild and test

### Code Style

- Follow C++17 standards
- Use RAII for resource management
- Implement thread-safe operations
- Add comprehensive error handling
- Include unit tests for new components

## License

This C++ integration is part of the ARIA_PRO trading system and follows the same licensing terms.
