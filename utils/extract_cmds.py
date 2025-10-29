"""
extract_cmds.py

This utility recursively traverses nested folders to find `.json` files,
extracts any "cmd" fields (from full JSON or line-delimited JSON),
and aggregates them into a single JSON output file.

- Input:  A root folder containing `.json` files (possibly nested).
- Output: A JSON file with a list of {"cmd": "..."} objects.

Example:
    python extract_cmds.py

Author: Florence
"""

import json
import os
from json.decoder import JSONDecodeError


def extract_cmd_from_nested_folders(root_folder, output_file):
    cmd_list = []
    processed_files = 0
    error_files = 0

    # Recursively traverse all subfolders
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()

                        # Try parsing as a whole JSON object
                        try:
                            data = json.loads(content)
                            if 'cmd' in data:
                                cmd_list.append({'cmd': data['cmd']})
                                processed_files += 1
                            continue
                        except JSONDecodeError:
                            pass

                        # Fallback: parse line by line (JSONL)
                        for line in content.split('\n'):
                            line = line.strip()
                            if line:
                                try:
                                    data = json.loads(line)
                                    if 'cmd' in data:
                                        cmd_list.append({'cmd': data['cmd']})
                                        processed_files += 1
                                except JSONDecodeError:
                                    pass

                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    error_files += 1

    # Save extracted commands to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cmd_list, f, ensure_ascii=False, indent=4)

    print(f"\nProcessing summary:")
    print(f"Total files processed: {processed_files + error_files}")
    print(f"Successfully extracted commands from: {processed_files} files")
    print(f"Files with errors: {error_files}")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    extract_cmd_from_nested_folders(
        "/Users/florence/Desktop/commands-master 2",
        "output.json"
    )
