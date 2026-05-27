// Dijkstra shortest path solver (C++17)
// Reads a weighted directed graph from CSV, computes shortest paths
// between named nodes, and outputs JSON results.

#include <algorithm>
#include <iomanip>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::istringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ','))
        out.push_back(trim(tok));
    return out;
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

struct Result {
    double distance = 0.0;
    bool reachable = false;
    std::vector<std::string> path;
};

Result dijkstra(const std::string& input_path,
                const std::string& source,
                const std::string& dest,
                bool directed) {
    std::unordered_map<std::string, std::vector<std::pair<std::string, double>>> graph;
    std::unordered_set<std::string> nodes;

    std::ifstream infile(input_path);
    if (!infile.is_open()) {
        std::cerr << "Cannot open " << input_path << "\n";
        return Result{};
    }

    std::string line;
    // Skip header
    if (!std::getline(infile, line)) {
        // empty file
    }
    while (std::getline(infile, line)) {
        std::string t = trim(line);
        if (t.empty() || t[0] == '#') continue;
        auto fields = split_csv(t);
        if (fields.size() < 3) continue;
        std::string src = fields[0], dst = fields[1];
        double w = std::stod(fields[2]);
        nodes.insert(src);
        nodes.insert(dst);
        graph[src].emplace_back(dst, w);
        if (!directed)
            graph[dst].emplace_back(src, w);
    }

    nodes.insert(source);
    nodes.insert(dest);

    std::unordered_map<std::string, double> dist;
    std::unordered_map<std::string, std::optional<std::string>> prev;
    for (auto& n : nodes) { dist[n] = 1e18; prev[n] = std::nullopt; }
    dist[source] = 0.0;

    using PQ = std::priority_queue<std::tuple<double, std::string>,
                                   std::vector<std::tuple<double, std::string>>,
                                   std::greater<>>;
    PQ pq;
    pq.emplace(0.0, source);

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        if (u == dest) break;
        auto it = graph.find(u);
        if (it == graph.end()) continue;
        for (auto& [v, w] : it->second) {
            double nd = d + w;
            if (nd < dist[v]) {
                dist[v] = nd;
                prev[v] = u;
                pq.emplace(nd, v);
            }
        }
    }

    Result r;
    if (dist[dest] >= 1e17) {
        r.reachable = false;
        r.distance = 1e18;
    } else {
        r.reachable = true;
        r.distance = dist[dest];
        std::string cur = dest;
        while (cur != source) {
            r.path.push_back(cur);
            auto it = prev.find(cur);
            if (it == prev.end() || !it->second.has_value()) break;
            cur = it->second.value();
            if (cur.empty()) break;
        }
        r.path.push_back(source);
        std::reverse(r.path.begin(), r.path.end());
    }
    return r;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: solve <graph.csv> <source> <dest> [--undirected]\n";
        return 1;
    }

    std::string input_path = argv[1];
    std::string source = argv[2];
    std::string dest = argv[3];
    bool directed = true;
    for (int i = 4; i < argc; i++) {
        if (std::string(argv[i]) == "--undirected") directed = false;
    }

    Result r = dijkstra(input_path, source, dest, directed);

    std::cout << "{";
    std::cout << "\"source\":\"" << escape_json_str(source) << "\",\n";
    std::cout << "\"destination\":\"" << escape_json_str(dest) << "\",\n";
    if (!r.reachable) {
        std::cout << "\"distance\":Infinity,\n";
        std::cout << "\"path\":null\n";
    } else {
       std::cout << "\"distance\":" << std::fixed << r.distance << ",\n";
        std::cout << "\"path\":[";
        for (size_t i = 0; i < r.path.size(); i++) {
            if (i) std::cout << ",";
            std::cout << "\"" << escape_json_str(r.path[i]) << "\"";
        }
        std::cout << "]\n";
    }
    std::cout << "}\n";
    return 0;
}
