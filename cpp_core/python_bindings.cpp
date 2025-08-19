#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "../include/market_data_processor.h"
#include "../include/smc_engine.h"

namespace py = pybind11;

PYBIND11_MODULE(aria_core, m) {
    m.doc() = "ARIA_PRO C++ Core - High-performance market data processing and SMC pattern detection";
    
    // Data structures
    py::class_<TickData>(m, "TickData")
        .def(py::init<>())
        .def_readwrite("symbol", &TickData::symbol)
        .def_readwrite("bid", &TickData::bid)
        .def_readwrite("ask", &TickData::ask)
        .def_readwrite("volume", &TickData::volume)
        .def_readwrite("timestamp", &TickData::timestamp);
    
    py::class_<BarData>(m, "BarData")
        .def(py::init<>())
        .def_readwrite("symbol", &BarData::symbol)
        .def_readwrite("ts", &BarData::ts)
        .def_readwrite("open", &BarData::open)
        .def_readwrite("high", &BarData::high)
        .def_readwrite("low", &BarData::low)
        .def_readwrite("close", &BarData::close)
        .def_readwrite("volume", &BarData::volume);
    
    py::class_<OrderBlock>(m, "OrderBlock")
        .def(py::init<>())
        .def_readwrite("high", &OrderBlock::high)
        .def_readwrite("low", &OrderBlock::low)
        .def_readwrite("open", &OrderBlock::open)
        .def_readwrite("close", &OrderBlock::close)
        .def_readwrite("timestamp", &OrderBlock::timestamp)
        .def_readwrite("is_bullish", &OrderBlock::is_bullish)
        .def_readwrite("is_mitigated", &OrderBlock::is_mitigated);
    
    py::class_<FairValueGap>(m, "FairValueGap")
        .def(py::init<>())
        .def_readwrite("high", &FairValueGap::high)
        .def_readwrite("low", &FairValueGap::low)
        .def_readwrite("timestamp", &FairValueGap::timestamp)
        .def_readwrite("is_bullish", &FairValueGap::is_bullish)
        .def_readwrite("is_filled", &FairValueGap::is_filled);
    
    py::class_<LiquidityLevel>(m, "LiquidityLevel")
        .def(py::init<>())
        .def_readwrite("price", &LiquidityLevel::price)
        .def_readwrite("timestamp", &LiquidityLevel::timestamp)
        .def_readwrite("is_bullish", &LiquidityLevel::is_bullish)
        .def_readwrite("volume", &LiquidityLevel::volume);
    
    py::class_<Pattern>(m, "Pattern")
        .def(py::init<>())
        .def_readwrite("type", &Pattern::type)
        .def_readwrite("entry_price", &Pattern::entry_price)
        .def_readwrite("stop_loss", &Pattern::stop_loss)
        .def_readwrite("take_profit", &Pattern::take_profit)
        .def_readwrite("confidence", &Pattern::confidence)
        .def_readwrite("timestamp", &Pattern::timestamp);
    
    // MarketDataProcessor class
    py::class_<MarketDataProcessor>(m, "MarketDataProcessor")
        .def(py::init<>())
        .def("process_tick", &MarketDataProcessor::process_tick)
        .def("process_bar", &MarketDataProcessor::process_bar)
        .def("calculate_sma", &MarketDataProcessor::calculate_sma)
        .def("calculate_ema", &MarketDataProcessor::calculate_ema)
        .def("calculate_rsi", &MarketDataProcessor::calculate_rsi)
        .def("detect_doji", &MarketDataProcessor::detect_doji)
        .def("detect_hammer", &MarketDataProcessor::detect_hammer)
        .def("detect_engulfing", &MarketDataProcessor::detect_engulfing)
        .def("start", &MarketDataProcessor::start)
        .def("stop", &MarketDataProcessor::stop)
        .def("is_running", &MarketDataProcessor::is_running)
        .def("get_recent_bars", &MarketDataProcessor::get_recent_bars)
        .def("get_latest_tick", &MarketDataProcessor::get_latest_tick);
    
    // SMCEngine class
    py::class_<SMCEngine>(m, "SMCEngine")
        .def(py::init<>())
        .def("detect_order_block", &SMCEngine::detect_order_block)
        .def("get_order_blocks", &SMCEngine::get_order_blocks)
        .def("is_order_block_mitigated", &SMCEngine::is_order_block_mitigated)
        .def("detect_fair_value_gap", &SMCEngine::detect_fair_value_gap)
        .def("get_fair_value_gaps", &SMCEngine::get_fair_value_gaps)
        .def("is_fvg_filled", &SMCEngine::is_fvg_filled)
        .def("detect_liquidity_levels", &SMCEngine::detect_liquidity_levels)
        .def("get_liquidity_levels", &SMCEngine::get_liquidity_levels)
        .def("detect_breakout_pattern", &SMCEngine::detect_breakout_pattern)
        .def("detect_reversal_pattern", &SMCEngine::detect_reversal_pattern)
        .def("generate_signals", &SMCEngine::generate_signals)
        .def("calculate_confidence", &SMCEngine::calculate_confidence)
        .def("clear_old_data", &SMCEngine::clear_old_data)
        .def("update_patterns", &SMCEngine::update_patterns);
    
    // Module-level functions for convenience
    m.def("create_market_processor", []() {
        return std::make_unique<MarketDataProcessor>();
    });
    
    m.def("create_smc_engine", []() {
        return std::make_unique<SMCEngine>();
    });
    
    // Version info
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "ARIA_PRO Team";
}
