// JSON-lines log analyzer (C++17)
// Parses a log.jsonl file, computes statistics on duration_ms,
// and outputs JSON results.

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static std::string extract_json_string(const std::string& s, const std::string& key) {
    std::string target = "\"" + key + "\"";
    size_t pos = s.find(target);
    if (pos == std::string::npos) return "";
    pos = s.find(':', pos + target.size());
    if (pos == std::string::npos) return "";
    pos++; // skip ':'
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t')) pos++;
    if (pos >= s.size() || s[pos] != '"') return "";
    pos++;
    std::string result;
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < s.size()) { pos++; }
        result += s[pos];
        pos++;
    }
    return result;
}

static double extract_json_number(const std::string& s, const std::string& key, double default_val = 0.0) {
    std::string target = "\"" + key + "\"";
    size_t pos = s.find(target);
    if (pos == std::string::npos) return default_val;
    pos = s.find(':', pos + target.size());
    if (pos == std::string::npos) return default_val;
    pos++;
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t')) pos++;
    try {
        return std::stod(s.substr(pos));
    } catch (...) {
        return default_val;
    }
}

static std::string escape_json_str(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else out += c;
    }
    return out;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: solve <file.jsonl> [--output <path>] [--level <L>] [--top-n <N>]\n";
        return 1;
    }

    std::string filepath = argv[1];
    std::string output_path;
    std::string filter_level;
    int top_n = 10;

    for (int i = 2; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--output" && i + 1 < argc) {
            output_path = argv[i + 1];
            i++;
        } else if (arg == "--level" && i + 1 < argc) {
            filter_level = argv[i + 1];
            i++;
        } else if (arg == "--top-n" && i + 1 < argc) {
            top_n = std::stoi(argv[i + 1]);
            i++;
        }
    }

    std::ifstream infile(filepath);
    if (!infile.is_open()) {
        std::cerr << "Cannot open " << filepath << "\n";
        return 1;
    }

    // Parse entries
    struct Entry {
        std::string level;
        std::string service;
        double duration_ms = 0;
    };
    std::vector<Entry> all_entries;
    std::string line;

    while (std::getline(infile, line)) {
        std::string t = trim(line);
        if (t.empty()) continue;
        Entry e;
        e.level = extract_json_string(t, "level");
        e.service = extract_json_string(t, "service");
        e.duration_ms = extract_json_number(t, "duration_ms", 0.0);
        all_entries.push_back(e);
    }

    // total_entries is total count in file (before any filtering)
    int total_entries = (int)all_entries.size();

    // Filter if requested
    std::vector<const Entry*> entries;
    for (auto& e : all_entries) {
        if (!filter_level.empty() && e.level != filter_level) continue;
        entries.push_back(&e);
    }

    // Level counts (of filtered entries)
    std::unordered_map<std::string, int> level_counts;
    for (auto* e : entries) level_counts[e->level]++;

    // Service counts (of filtered entries)
    std::unordered_map<std::string, int> service_counts;
    for (auto* e : entries) service_counts[e->service]++;

    // Per-service error counts (of filtered entries)
    std::unordered_map<std::string, int> per_service_errors;
    for (auto* e : entries) {
        if (e->level == "ERROR") per_service_errors[e->service]++;
    }

    // Sort by duration descending
    std::vector<const Entry*> by_duration(entries.begin(), entries.end());
    std::sort(by_duration.begin(), by_duration.end(),
              [](const Entry* a, const Entry* b) { return a->duration_ms > b->duration_ms; });

    // Build JSON output
    std::ostringstream out;
    out << "{\n";
    out << "  \"total_entries\": " << total_entries << ",\n";

    // level_counts
    out << "  \"level_counts\": {";
    {
        std::vector<std::string> keys;
        for (auto& [k, v] : level_counts) keys.push_back(k);
        std::sort(keys.begin(), keys.end());
        for (size_t i = 0; i < keys.size(); i++) {
            if (i) out << ", ";
            out << "\"" << escape_json_str(keys[i]) << "\": " << level_counts[keys[i]];
        }
    }
    out << "},\n";

    // service_counts
    out << "  \"service_counts\": {";
    {
        std::vector<std::string> keys;
        for (auto& [k, v] : service_counts) keys.push_back(k);
        std::sort(keys.begin(), keys.end());
        for (size_t i = 0; i < keys.size(); i++) {
            if (i) out << ", ";
            out << "\"" << escape_json_str(keys[i]) << "\": " << service_counts[keys[i]];
        }
    }
    out << "},\n";

    // per_service_error_counts
    out << "  \"per_service_error_counts\": {";
    {
        std::vector<std::string> keys;
        for (auto& [k, v] : per_service_errors) keys.push_back(k);
        std::sort(keys.begin(), keys.end());
        for (size_t i = 0; i < keys.size(); i++) {
            if (i) out << ", ";
            out << "\"" << escape_json_str(keys[i]) << "\": " << per_service_errors[keys[i]];
        }
    }
    out << "},\n";

    // slowest_entries
    out << "  \"slowest_entries\": [\n";
    int limit = std::min(top_n, (int)by_duration.size());
    for (int i = 0; i < limit; i++) {
        out << "    {\"duration_ms\": " << by_duration[i]->duration_ms
            << ", \"level\": \"" << escape_json_str(by_duration[i]->level)
            << "\", \"service\": \"" << escape_json_str(by_duration[i]->service) << "\"}";
        if (i + 1 < limit) out << ",";
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";

    std::string json_out = out.str();

    if (!output_path.empty()) {
        std::ofstream out_file(output_path);
        out_file << json_out;
    } else {
        std::cout << json_out;
    }

    return 0;
}
