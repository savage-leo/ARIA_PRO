#pragma once
#include <vector>
#include <string>
#include <memory>

// SMC-specific data structures
struct OrderBlock {
    double high;
    double low;
    double open;
    double close;
    uint64_t timestamp;
    bool is_bullish;
    bool is_mitigated;
};

struct FairValueGap {
    double high;
    double low;
    uint64_t timestamp;
    bool is_bullish;
    bool is_filled;
};

struct LiquidityLevel {
    double price;
    uint64_t timestamp;
    bool is_bullish;
    double volume;
};

struct Pattern {
    std::string type;
    double entry_price;
    double stop_loss;
    double take_profit;
    double confidence;
    uint64_t timestamp;
};

// Smart Money Concepts Engine
class SMCEngine {
private:
    std::vector<OrderBlock> order_blocks_;
    std::vector<FairValueGap> fair_value_gaps_;
    std::vector<LiquidityLevel> liquidity_levels_;
    std::vector<Pattern> patterns_;
    
public:
    SMCEngine();
    ~SMCEngine();
    
    // Order Block detection
    bool detect_order_block(double open, double high, double low, double close, uint64_t timestamp);
    std::vector<OrderBlock> get_order_blocks(const std::string& symbol);
    bool is_order_block_mitigated(const OrderBlock& block, double current_price);
    
    // Fair Value Gap detection
    bool detect_fair_value_gap(double prev_high, double prev_low, 
                              double curr_high, double curr_low, uint64_t timestamp);
    std::vector<FairValueGap> get_fair_value_gaps(const std::string& symbol);
    bool is_fvg_filled(const FairValueGap& fvg, double current_price);
    
    // Liquidity detection
    void detect_liquidity_levels(const std::vector<double>& prices, 
                                const std::vector<double>& volumes, 
                                const std::vector<uint64_t>& timestamps);
    std::vector<LiquidityLevel> get_liquidity_levels(const std::string& symbol);
    
    // Pattern recognition
    Pattern detect_breakout_pattern(const std::vector<double>& prices, 
                                   const std::vector<double>& volumes);
    Pattern detect_reversal_pattern(const std::vector<double>& prices, 
                                   const std::vector<double>& volumes);
    
    // Trading signals
    std::vector<Pattern> generate_signals(const std::string& symbol);
    double calculate_confidence(const Pattern& pattern);
    
    // Data management
    void clear_old_data(uint64_t cutoff_timestamp);
    void update_patterns();
};
