// Markdown TOC generator (C++17)
// Extracts headings from a file and outputs JSON structure.

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// Extract heading level (number of # at start) and title
static std::optional<std::pair<int, std::string>> try_parse_heading(const std::string& line, int max_depth) {
    int lvl = 0;
    for (char c : line) {
        if (c == '#') {
            lvl++;
            if (lvl > max_depth) return std::nullopt;
        } else if (c == ' ' || c == '\t') {
            break;
        } else {
            return std::nullopt; // not a heading
        }
    }
    if (lvl == 0) return std::nullopt;
    // Find title after #... followed by space
    size_t pos = lvl;
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) pos++;
    if (pos >= line.size()) return std::nullopt;
    return std::make_pair(lvl, trim(line.substr(pos)));
}

struct Node {
    std::string title;
    int level = 0;
    std::vector<Node> children;
};

static std::string escape_json_str(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else out += c;
    }
    return out;
}

static std::string node_to_json(const Node& n, int indent) {
    std::string pad(indent, ' ');
    std::string pad2(indent + 2, ' ');
    std::string result = pad + "{\n";
    result += pad2 + "\"title\": \"" + escape_json_str(n.title) + "\",\n";
    result += pad2 + "\"level\": " + std::to_string(n.level) + ",\n";
    if (n.children.empty()) {
        result += pad2 + "\"children\": []\n";
    } else {
        result += pad2 + "\"children\": [\n";
        for (size_t i = 0; i < n.children.size(); i++) {
            result += node_to_json(n.children[i], indent + 4);
            if (i + 1 < n.children.size()) result += ",";
            result += "\n";
        }
        result += pad2 + "]\n";
    }
    result += pad + "}";
    return result;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: solve <file.md> [--output <path>] [--max-depth <n>]\n";
        return 1;
    }

    std::string filepath = argv[1];
    std::string output_path;
    int max_depth = 999;

    for (int i = 2; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--output" && i + 1 < argc) {
            output_path = argv[i + 1];
            i++;
        } else if (arg == "--max-depth" && i + 1 < argc) {
            max_depth = std::stoi(argv[i + 1]);
            i++;
        }
    }

    std::ifstream infile(filepath);
    if (!infile.is_open()) {
        std::cerr << "Cannot open " << filepath << "\n";
        return 1;
    }

    // Parse headings
    std::vector<std::pair<int, std::string>> headings;
    std::string line;
    while (std::getline(infile, line)) {
        auto h = try_parse_heading(trim(line), max_depth);
        if (h) headings.push_back(*h);
    }

    // Build nested TOC: skip H1 as item, H2+ as top-level
    std::vector<Node> root;
    std::vector<std::pair<int, Node*>> stack;

    for (auto& [lvl, title] : headings) {
        Node node{title, lvl, {}};

        if (lvl == 1) {
            // H1 is the title, skip as an item
            continue;
        }

        // Pop stack until we find a parent with lower level
        while (!stack.empty() && stack.back().first >= lvl)
            stack.pop_back();

        if (!stack.empty())
            stack.back().second->children.push_back(node);
        else
            root.push_back(node);

        stack.emplace_back(lvl, &root.back());
    }

    // Output
    std::string json_out = "[\n";
    for (size_t i = 0; i < root.size(); i++) {
        json_out += node_to_json(root[i], 2);
        if (i + 1 < root.size()) json_out += ",";
        json_out += "\n";
    }
    json_out += "]\n";

    if (!output_path.empty()) {
        std::ofstream out(output_path);
        out << json_out;
    } else {
        std::cout << json_out;
    }

    return 0;
}
