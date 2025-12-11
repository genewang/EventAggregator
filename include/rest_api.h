#pragma once

#include "event_aggregator.h"
#include <string>
#include <memory>
#include <functional>

namespace event_aggregator {

// Simple REST API server for dashboard integration
// Uses a callback-based approach for flexibility
class RestApi {
public:
    using QueryHandler = std::function<QueryResult(const QueryParams&)>;
    using StatsHandler = std::function<std::string()>;
    
    RestApi(int port = 8080);
    ~RestApi();
    
    // Register handlers
    void set_query_handler(QueryHandler handler);
    void set_stats_handler(StatsHandler handler);
    
    // Start server (blocking)
    void run();
    
    // Stop server
    void stop();
    
private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// Helper: Parse query parameters from HTTP request
QueryParams parse_query_params(const std::string& query_string);

} // namespace event_aggregator

