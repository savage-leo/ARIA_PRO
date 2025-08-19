#include "../include/smc_engine.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>

SMCEngine::SMCEngine() = default;
SMCEngine::~SMCEngine() = default;

bool SMCEngine::detect_order_block(double open, double high, double low, double close, uint64_t timestamp) {
    double body_size = std::abs(close - open);
    double total_range = high - low;
    
    if (total_range == 0.0) return false;
    
    double body_ratio = body_size / total_range;
    bool is_bullish = close > open;
    
    // Order block criteria: strong move with small body relative to range
    if (body_ratio > 0.6 && body_ratio < 0.9) {
        OrderBlock block;
        block.high = high;
        block.low = low;
        block.open = open;
        block.close = close;
        block.timestamp = timestamp;
        block.is_bullish = is_bullish;
        block.is_mitigated = false;
        
        order_blocks_.push_back(block);
        
        // Keep only recent order blocks (last 100)
        if (order_blocks_.size() > 100) {
            order_blocks_.erase(order_blocks_.begin());
        }
        
        return true;
    }
    
    return false;
}

std::vector<OrderBlock> SMCEngine::get_order_blocks(const std::string& symbol) {
    // For now, return all order blocks. In a real implementation,
    // you'd filter by symbol
    return order_blocks_;
}

bool SMCEngine::is_order_block_mitigated(const OrderBlock& block, double current_price) {
    if (block.is_bullish) {
        // Bullish order block is mitigated when price goes below the low
        return current_price < block.low;
    } else {
        // Bearish order block is mitigated when price goes above the high
        return current_price > block.high;
    }
}

bool SMCEngine::detect_fair_value_gap(double prev_high, double prev_low, 
                                     double curr_high, double curr_low, uint64_t timestamp) {
    // Bullish FVG: previous low > current high
    if (prev_low > curr_high) {
        FairValueGap fvg;
        fvg.high = prev_low;
        fvg.low = curr_high;
        fvg.timestamp = timestamp;
        fvg.is_bullish = true;
        fvg.is_filled = false;
        
        fair_value_gaps_.push_back(fvg);
        return true;
    }
    
    // Bearish FVG: previous high < current low
    if (prev_high < curr_low) {
        FairValueGap fvg;
        fvg.high = curr_low;
        fvg.low = prev_high;
        fvg.timestamp = timestamp;
        fvg.is_bullish = false;
        fvg.is_filled = false;
        
        fair_value_gaps_.push_back(fvg);
        return true;
    }
    
    return false;
}

std::vector<FairValueGap> SMCEngine::get_fair_value_gaps(const std::string& symbol) {
    // For now, return all FVGs. In a real implementation,
    // you'd filter by symbol
    return fair_value_gaps_;
}

bool SMCEngine::is_fvg_filled(const FairValueGap& fvg, double current_price) {
    if (fvg.is_bullish) {
        // Bullish FVG is filled when price goes below the low
        return current_price < fvg.low;
    } else {
        // Bearish FVG is filled when price goes above the high
        return current_price > fvg.high;
    }
}

void SMCEngine::detect_liquidity_levels(const std::vector<double>& prices, 
                                       const std::vector<double>& volumes, 
                                       const std::vector<uint64_t>& timestamps) {
    if (prices.size() != volumes.size() || prices.size() != timestamps.size()) {
        return;
    }
    
    // Simple liquidity detection: high volume levels
    double avg_volume = 0.0;
    for (double volume : volumes) {
        avg_volume += volume;
    }
    avg_volume /= volumes.size();
    
    for (size_t i = 0; i < prices.size(); ++i) {
        if (volumes[i] > avg_volume * 1.5) { // 50% above average
            LiquidityLevel level;
            level.price = prices[i];
            level.timestamp = timestamps[i];
            level.volume = volumes[i];
            level.is_bullish = (i > 0 && prices[i] > prices[i-1]);
            
            liquidity_levels_.push_back(level);
        }
    }
    
    // Keep only recent liquidity levels (last 50)
    if (liquidity_levels_.size() > 50) {
        liquidity_levels_.erase(liquidity_levels_.begin(), 
                               liquidity_levels_.begin() + (liquidity_levels_.size() - 50));
    }
}

std::vector<LiquidityLevel> SMCEngine::get_liquidity_levels(const std::string& symbol) {
    // For now, return all liquidity levels. In a real implementation,
    // you'd filter by symbol
    return liquidity_levels_;
}

Pattern SMCEngine::detect_breakout_pattern(const std::vector<double>& prices, 
                                          const std::vector<double>& volumes) {
    Pattern pattern;
    pattern.type = "breakout";
    pattern.confidence = 0.0;
    
    if (prices.size() < 20) return pattern;
    
    // Simple breakout detection: price breaks above recent high with volume
    double recent_high = *std::max_element(prices.begin() + prices.size() - 20, prices.end() - 1);
    double current_price = prices.back();
    double current_volume = volumes.back();
    
    double avg_volume = 0.0;
    for (size_t i = prices.size() - 20; i < prices.size() - 1; ++i) {
        avg_volume += volumes[i];
    }
    avg_volume /= 19;
    
    if (current_price > recent_high && current_volume > avg_volume * 1.2) {
        pattern.entry_price = current_price;
        pattern.stop_loss = recent_high * 0.995; // 0.5% below breakout level
        pattern.take_profit = current_price + (current_price - pattern.stop_loss) * 2; // 2:1 R:R
        pattern.confidence = std::min(0.9, current_volume / avg_volume / 2.0);
        pattern.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
    
    return pattern;
}

Pattern SMCEngine::detect_reversal_pattern(const std::vector<double>& prices, 
                                          const std::vector<double>& volumes) {
    Pattern pattern;
    pattern.type = "reversal";
    pattern.confidence = 0.0;
    
    if (prices.size() < 10) return pattern;
    
    // Simple reversal detection: price reversal with volume confirmation
    double recent_trend = 0.0;
    for (size_t i = prices.size() - 10; i < prices.size() - 1; ++i) {
        recent_trend += (prices[i+1] - prices[i]);
    }
    
    double current_price = prices.back();
    double current_volume = volumes.back();
    
    double avg_volume = 0.0;
    for (size_t i = prices.size() - 10; i < prices.size() - 1; ++i) {
        avg_volume += volumes[i];
    }
    avg_volume /= 9;
    
    // Bullish reversal after downtrend
    if (recent_trend < 0 && current_price > prices[prices.size() - 2] && current_volume > avg_volume) {
        pattern.entry_price = current_price;
        pattern.stop_loss = *std::min_element(prices.begin() + prices.size() - 10, prices.end());
        pattern.take_profit = current_price + (current_price - pattern.stop_loss) * 1.5;
        pattern.confidence = std::min(0.8, current_volume / avg_volume / 1.5);
        pattern.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
    // Bearish reversal after uptrend
    else if (recent_trend > 0 && current_price < prices[prices.size() - 2] && current_volume > avg_volume) {
        pattern.entry_price = current_price;
        pattern.stop_loss = *std::max_element(prices.begin() + prices.size() - 10, prices.end());
        pattern.take_profit = current_price - (pattern.stop_loss - current_price) * 1.5;
        pattern.confidence = std::min(0.8, current_volume / avg_volume / 1.5);
        pattern.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
    
    return pattern;
}

std::vector<Pattern> SMCEngine::generate_signals(const std::string& symbol) {
    std::vector<Pattern> signals;
    
    // Generate signals based on order blocks and FVGs
    for (const auto& block : order_blocks_) {
        if (!block.is_mitigated) {
            Pattern signal;
            signal.type = block.is_bullish ? "buy" : "sell";
            signal.entry_price = block.is_bullish ? block.high : block.low;
            signal.stop_loss = block.is_bullish ? block.low : block.high;
            signal.take_profit = block.is_bullish ? 
                signal.entry_price + (signal.entry_price - signal.stop_loss) * 2 :
                signal.entry_price - (signal.stop_loss - signal.entry_price) * 2;
            signal.confidence = 0.7;
            signal.timestamp = block.timestamp;
            
            signals.push_back(signal);
        }
    }
    
    return signals;
}

double SMCEngine::calculate_confidence(const Pattern& pattern) {
    // Simple confidence calculation based on pattern type
    if (pattern.type == "breakout") {
        return std::min(0.9, pattern.confidence);
    } else if (pattern.type == "reversal") {
        return std::min(0.8, pattern.confidence);
    } else if (pattern.type == "buy" || pattern.type == "sell") {
        return std::min(0.7, pattern.confidence);
    }
    
    return 0.5;
}

void SMCEngine::clear_old_data(uint64_t cutoff_timestamp) {
    // Remove old order blocks
    order_blocks_.erase(
        std::remove_if(order_blocks_.begin(), order_blocks_.end(),
                      [cutoff_timestamp](const OrderBlock& block) {
                          return block.timestamp < cutoff_timestamp;
                      }),
        order_blocks_.end()
    );
    
    // Remove old FVGs
    fair_value_gaps_.erase(
        std::remove_if(fair_value_gaps_.begin(), fair_value_gaps_.end(),
                      [cutoff_timestamp](const FairValueGap& fvg) {
                          return fvg.timestamp < cutoff_timestamp;
                      }),
        fair_value_gaps_.end()
    );
    
    // Remove old liquidity levels
    liquidity_levels_.erase(
        std::remove_if(liquidity_levels_.begin(), liquidity_levels_.end(),
                      [cutoff_timestamp](const LiquidityLevel& level) {
                          return level.timestamp < cutoff_timestamp;
                      }),
        liquidity_levels_.end()
    );
}

void SMCEngine::update_patterns() {
    // Update pattern status based on current market conditions
    // This would typically be called with current price data
    patterns_.clear();
    
    // Generate new patterns based on current market state
    // This is a simplified implementation
}
