#include "../include/market_data_processor.h"
#include <algorithm>
#include <cmath>
#include <deque>
#include <unordered_map>
#include <chrono>
#include <memory>
#include <atomic>

// LockFreeQueue implementation
template<typename T>
LockFreeQueue<T>::LockFreeQueue() {
    Node* dummy = new Node(T{});
    head_.store(dummy);
    tail_.store(dummy);
}

template<typename T>
LockFreeQueue<T>::~LockFreeQueue() {
    Node* current = head_.load();
    while (current) {
        Node* next = current->next.load();
        delete current;
        current = next;
    }
}

template<typename T>
void LockFreeQueue<T>::push(const T& data) {
    Node* new_node = new Node(data);
    Node* tail;
    Node* next;
    
    while (true) {
        tail = tail_.load();
        next = tail->next.load();
        
        if (tail == tail_.load()) {
            if (next == nullptr) {
                if (tail->next.compare_exchange_weak(next, new_node)) {
                    break;
                }
            } else {
                tail_.compare_exchange_weak(tail, next);
            }
        }
    }
    tail_.compare_exchange_weak(tail, new_node);
}

template<typename T>
bool LockFreeQueue<T>::pop(T& data) {
    Node* head;
    Node* tail;
    Node* next;
    
    while (true) {
        head = head_.load();
        tail = tail_.load();
        next = head->next.load();
        
        if (head == head_.load()) {
            if (head == tail) {
                if (next == nullptr) {
                    return false;
                }
                tail_.compare_exchange_weak(tail, next);
            } else {
                if (next == nullptr) {
                    continue;
                }
                data = next->data;
                if (head_.compare_exchange_weak(head, next)) {
                    break;
                }
            }
        }
    }
    delete head;
    return true;
}

template<typename T>
bool LockFreeQueue<T>::empty() const {
    return head_.load() == tail_.load();
}

// MarketDataProcessor implementation
MarketDataProcessor::MarketDataProcessor() 
    : tick_queue_(std::make_unique<LockFreeQueue<TickData>>()), running_(false) {
}

MarketDataProcessor::~MarketDataProcessor() = default;

void MarketDataProcessor::process_tick(const TickData& tick) {
    if (!running_) return;
    
    tick_queue_->push(tick);
    
    // Process high-frequency indicators
    calculate_sma(tick.symbol, 20);
    calculate_ema(tick.symbol, 20);
    calculate_rsi(tick.symbol, 14);
}

void MarketDataProcessor::process_bar(const BarData& bar) {
    if (!running_) return;
    
    // Store bar data
    bars_.push_back(bar);
    
    // Keep only recent bars (last 1000)
    if (bars_.size() > 1000) {
        bars_.erase(bars_.begin());
    }
    
    // Detect patterns
    if (bars_.size() >= 2) {
        detect_doji(bar);
        detect_hammer(bar);
        if (bars_.size() >= 2) {
            detect_engulfing(bars_[bars_.size() - 2], bar);
        }
    }
}

double MarketDataProcessor::calculate_sma(const std::string& symbol, int period) {
    if (bars_.size() < period) return 0.0;
    
    double sum = 0.0;
    int count = 0;
    
    for (int i = bars_.size() - 1; i >= 0 && count < period; --i) {
        if (bars_[i].symbol == symbol) {
            sum += bars_[i].close;
            count++;
        }
    }
    
    return count > 0 ? sum / count : 0.0;
}

double MarketDataProcessor::calculate_ema(const std::string& symbol, int period) {
    if (bars_.size() < period) return 0.0;
    
    double multiplier = 2.0 / (period + 1.0);
    double ema = 0.0;
    int count = 0;
    
    // Calculate initial SMA
    for (int i = bars_.size() - 1; i >= 0 && count < period; --i) {
        if (bars_[i].symbol == symbol) {
            ema += bars_[i].close;
            count++;
        }
    }
    
    if (count == 0) return 0.0;
    ema /= count;
    
    // Calculate EMA
    for (int i = bars_.size() - period - 1; i >= 0; --i) {
        if (bars_[i].symbol == symbol) {
            ema = (bars_[i].close * multiplier) + (ema * (1 - multiplier));
        }
    }
    
    return ema;
}

double MarketDataProcessor::calculate_rsi(const std::string& symbol, int period) {
    if (bars_.size() < period + 1) return 0.0;
    
    std::vector<double> gains, losses;
    
    for (int i = bars_.size() - 1; i > 0; --i) {
        if (bars_[i].symbol == symbol && bars_[i-1].symbol == symbol) {
            double change = bars_[i].close - bars_[i-1].close;
            if (change > 0) {
                gains.push_back(change);
                losses.push_back(0.0);
            } else {
                gains.push_back(0.0);
                losses.push_back(-change);
            }
        }
    }
    
    if (gains.size() < period) return 0.0;
    
    double avg_gain = 0.0, avg_loss = 0.0;
    for (int i = 0; i < period; ++i) {
        avg_gain += gains[i];
        avg_loss += losses[i];
    }
    avg_gain /= period;
    avg_loss /= period;
    
    if (avg_loss == 0.0) return 100.0;
    
    double rs = avg_gain / avg_loss;
    return 100.0 - (100.0 / (1.0 + rs));
}

bool MarketDataProcessor::detect_doji(const BarData& bar) {
    double body_size = std::abs(bar.close - bar.open);
    double total_range = bar.high - bar.low;
    
    if (total_range == 0.0) return false;
    
    double body_ratio = body_size / total_range;
    return body_ratio < 0.1; // Doji if body is less than 10% of total range
}

bool MarketDataProcessor::detect_hammer(const BarData& bar) {
    double body_size = std::abs(bar.close - bar.open);
    double total_range = bar.high - bar.low;
    
    if (total_range == 0.0) return false;
    
    double body_ratio = body_size / total_range;
    double lower_shadow = std::min(bar.open, bar.close) - bar.low;
    double upper_shadow = bar.high - std::max(bar.open, bar.close);
    
    return body_ratio < 0.3 && lower_shadow > 2 * body_size && upper_shadow < body_size;
}

bool MarketDataProcessor::detect_engulfing(const BarData& prev, const BarData& curr) {
    double prev_body = std::abs(prev.close - prev.open);
    double curr_body = std::abs(curr.close - curr.open);
    
    bool prev_bullish = prev.close > prev.open;
    bool curr_bullish = curr.close > curr.open;
    
    // Bullish engulfing
    if (!prev_bullish && curr_bullish) {
        return curr.open < prev.close && curr.close > prev.open && curr_body > prev_body;
    }
    
    // Bearish engulfing
    if (prev_bullish && !curr_bullish) {
        return curr.open > prev.close && curr.close < prev.open && curr_body > prev_body;
    }
    
    return false;
}

void MarketDataProcessor::start() {
    running_ = true;
}

void MarketDataProcessor::stop() {
    running_ = false;
}

bool MarketDataProcessor::is_running() const {
    return running_;
}

std::vector<BarData> MarketDataProcessor::get_recent_bars(const std::string& symbol, int count) {
    std::vector<BarData> result;
    for (int i = bars_.size() - 1; i >= 0 && result.size() < count; --i) {
        if (bars_[i].symbol == symbol) {
            result.insert(result.begin(), bars_[i]);
        }
    }
    return result;
}

TickData MarketDataProcessor::get_latest_tick(const std::string& symbol) {
    TickData latest_tick;
    TickData temp_tick;
    
    // Find the latest tick for the symbol
    while (tick_queue_->pop(temp_tick)) {
        if (temp_tick.symbol == symbol) {
            latest_tick = temp_tick;
        }
    }
    
    return latest_tick;
}
