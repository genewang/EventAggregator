// query_api.cpp
// Query API implementation and helper functions

#include "event_aggregator.h"
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace event_aggregator {

// Helper function to convert QueryResult to JSON (for REST API)
std::string QueryResult::to_json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    
    oss << "{\n";
    oss << "  \"region\": " << region << ",\n";
    oss << "  \"device\": " << device << ",\n";
    oss << "  \"node_name\": \"" << node_name << "\",\n";
    oss << "  \"start_ts\": " << start_ts << ",\n";
    oss << "  \"end_ts\": " << end_ts << ",\n";
    oss << "  \"total_count\": " << total_count << ",\n";
    oss << "  \"avg_gpu_util\": " << avg_gpu_util << ",\n";
    oss << "  \"max_power_actual\": " << max_power_actual << ",\n";
    oss << "  \"max_power_cap\": " << max_power_cap << ",\n";
    oss << "  \"min_power_cap\": " << min_power_cap << ",\n";
    oss << "  \"bins\": [\n";
    
    for (size_t i = 0; i < bins.size(); ++i) {
        oss << "    {\n";
        oss << "      \"timestamp\": " << bin_timestamps[i] << ",\n";
        oss << "      \"count\": " << bins[i].count << ",\n";
        oss << "      \"avg_util\": " 
            << (bins[i].count > 0 ? bins[i].sum_util / bins[i].count : 0.0) << ",\n";
        oss << "      \"max_power_actual\": " << bins[i].max_power_actual << ",\n";
        oss << "      \"max_power_cap\": " << bins[i].max_power_cap << ",\n";
        oss << "      \"min_power_cap\": " << bins[i].min_power_cap << "\n";
        oss << "    }";
        if (i < bins.size() - 1) oss << ",";
        oss << "\n";
    }
    
    oss << "  ]\n";
    oss << "}";
    
    return oss.str();
}

} // namespace event_aggregator

