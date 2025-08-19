#pragma once

#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <chrono>
#include <unordered_map>

// Forward declarations
struct TickData;
struct BarData;

// Simple lock-free queue implementation
template<typename T>
class LockFreeQueue {
private:
    struct Node {
        T data;
        std::atomic<Node*> next;
        Node(const T& d) : data(d), next(nullptr) {}
    };
    
    std::atomic<Node*> head_;
    std::atomic<Node*> tail_;

public:
    LockFreeQueue();
    ~LockFreeQueue();
    void push(const T& data);
    bool pop(T& data);
    bool empty() const;
};

// Market data structures
struct TickData {
    std::string symbol;
    double bid;
    double ask;
    double volume;
    std::chrono::system_clock::time_point timestamp;
};

struct BarData {
    std::string symbol;
    std::chrono::system_clock::time_point ts;
    double open;
    double close;
    double high;
    double low;
    double volume;
};

// Market data processor class
class MarketDataProcessor {
private:
    std::unique_ptr<LockFreeQueue<TickData>> tick_queue_;
    std::atomic<bool> running_;
    std::unordered_map<std::string, std::vector<BarData>> bar_data_;

public:
    MarketDataProcessor();
    ~MarketDataProcessor();
    
    void process_tick(const TickData& tick);
    void process_bar(const BarData& bar);
    
    // Technical indicators
    double calculate_sma(const std::string& symbol, int period);
    double calculate_ema(const std::string& symbol, int period);
    double calculate_rsi(const std::string& symbol, int period);
    
    // Pattern detection
    bool detect_doji(const BarData& bar);
    bool detect_hammer(const BarData& bar);
    bool detect_engulfing(const BarData& prev, const BarData& curr);
    
    // Control methods
    void start();
    void stop();
    bool is_running() const;
    
    // Data access
    std::vector<BarData> get_recent_bars(const std::string& symbol, int count = 100);
    TickData get_latest_tick(const std::string& symbol);
};


