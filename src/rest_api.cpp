// rest_api.cpp
// REST API implementation using a simple HTTP server
// For production, consider using a proper HTTP library (e.g., httplib, crow, etc.)

#include "rest_api.h"
#include <sstream>
#include <thread>
#include <atomic>
#include <map>
#include <regex>

// Simple HTTP server implementation
// Note: This is a minimal implementation. For production, use a proper HTTP library.

namespace event_aggregator {

class RestApi::Impl {
public:
    int port_;
    std::atomic<bool> running_{false};
    std::thread server_thread_;
    QueryHandler query_handler_;
    StatsHandler stats_handler_;
    
    Impl(int port) : port_(port) {}
    
    void run_server() {
        // Simplified: In production, use a proper HTTP library
        // This is a placeholder showing the API structure
        std::cout << "REST API server would run on port " << port_ << std::endl;
        std::cout << "For production, integrate with httplib, crow, or similar" << std::endl;
    }
};

RestApi::RestApi(int port) : pimpl_(std::make_unique<Impl>(port)) {
}

RestApi::~RestApi() {
    stop();
}

void RestApi::set_query_handler(QueryHandler handler) {
    pimpl_->query_handler_ = std::move(handler);
}

void RestApi::set_stats_handler(StatsHandler handler) {
    pimpl_->stats_handler_ = std::move(handler);
}

void RestApi::run() {
    pimpl_->running_ = true;
    pimpl_->server_thread_ = std::thread([this]() {
        pimpl_->run_server();
    });
}

void RestApi::stop() {
    if (pimpl_->running_) {
        pimpl_->running_ = false;
        if (pimpl_->server_thread_.joinable()) {
            pimpl_->server_thread_.join();
        }
    }
}

QueryParams parse_query_params(const std::string& query_string) {
    QueryParams params;
    
    // Simple parsing (in production, use proper URL parsing)
    std::regex param_regex(R"((\w+)=([^&]+))");
    std::sregex_iterator iter(query_string.begin(), query_string.end(), param_regex);
    std::sregex_iterator end;
    
    for (; iter != end; ++iter) {
        std::smatch match = *iter;
        std::string key = match[1].str();
        std::string value = match[2].str();
        
        if (key == "region") {
            params.region = std::stoi(value);
        } else if (key == "device") {
            params.device = std::stoi(value);
        } else if (key == "node_name") {
            params.node_name = value;
        } else if (key == "start_ts") {
            params.start_ts = std::stoull(value);
        } else if (key == "end_ts") {
            params.end_ts = std::stoull(value);
        } else if (key == "bin_size") {
            params.bin_size_seconds = std::stoi(value);
        }
    }
    
    return params;
}

} // namespace event_aggregator

