#!/usr/bin/env python3
"""Markdown TOC generator.
Extracts headings from a markdown file and produces a structured TOC.
"""
import argparse
import json
import os
import re
import sys


def extract_headings(filepath: str, max_depth: int = 999):
    """Extract headings up to *max_depth* from a markdown file."""
    headings = []
    with open(filepath, "r") as f:
        for line in f:
            m = re.match(r'^(#{1,' + str(max_depth) + r'})\s+(.+)$', line)
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
                headings.append((level, title))
    return headings


def build_toc(headings, max_depth=999):
    """Build flat list of TOC items with children for nested structure.

    H2+ headings become top-level items. H1 is treated as title only
    and does not appear as an item (H2s are the first items).
    H3+ become children of their nearest H1/H2 ancestor.
    """
    # Separate H1 as title (not an item)
    root = []
    h1_title = None
    stack = []  # (level, node)

    for level, title in headings:
        if level > max_depth:
            continue

        node = {"title": title, "level": level, "children": []}

        if level == 1:
            h1_title = title
            continue  # skip H1 as an item

        # Pop stack until we find a parent with lower level
        while stack and stack[-1][0] >= level:
            stack.pop()

        if stack:
            stack[-1][1]["children"].append(node)
        else:
            root.append(node)

        stack.append((level, node))

    return root


def toc_to_html(toc, min_level=1):
    """Convert TOC list to HTML string."""
    parts = []
    for item in toc:
        level = item["level"]
        indent = "  " * (level - min_level)
        anchor = item["title"].lower().replace(" ", "-")
        parts.append(f'{indent}<li><a href="#{anchor}">{item["title"]}</a>')
        if item["children"]:
            parts.append(f'{indent}<ul>')
            parts.append(toc_to_html(item["children"], level + 1))
            parts.append(f'{indent}</ul>')
        parts.append(f'{indent}</li>')
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Generate a TOC from a markdown file.")
    parser.add_argument("file", help="Path to markdown file")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--max-depth", "-d", type=int, default=999, help="Max heading depth")
    parser.add_argument("--format", "-f", choices=["json", "html"], default="json", help="Output format")
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"Error: {args.file} is not a file", file=sys.stderr)
        sys.exit(1)

    headings = extract_headings(args.file, args.max_depth)
    toc = build_toc(headings, args.max_depth)

    if args.format == "html":
        output = "<!DOCTYPE html>\n<html>\n<head><title>Table of Contents</title></head>\n<body>\n<ul>\n"
        output += toc_to_html(toc)
        output += "\n</ul>\n</body>\n</html>\n"
    else:
        output = json.dumps(toc, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
