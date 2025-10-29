"""
json_to_csv.py

This utility converts a JSON file containing {"cmd": "..."} objects
into a deduplicated single-column CSV file.

- Input:  JSON file (array of {"cmd": "..."} objects).
- Output: CSV file with one unique command per line (no header).

Features:
- Strict deduplication (ignores whitespace differences).
- Skips entries without a "cmd" field.
- Reports number of duplicates removed.

Example:
    python json_to_csv.py

Author: Florence
"""

import json
import csv


def json_to_cmd_csv(json_file_path, csv_file_path):
    try:
        # Read JSON file
        with open(json_file_path, 'r', encoding='utf-8') as jsonfile:
            data = json.load(jsonfile)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain an array of objects")

        seen_commands = set()
        unique_commands = []
        duplicate_count = 0

        for item in data:
            if 'cmd' in item:
                cmd = item['cmd'].strip()
                if cmd not in seen_commands:
                    seen_commands.add(cmd)
                    unique_commands.append(cmd)
                else:
                    duplicate_count += 1
            else:
                print(f"Warning: skipped entry without 'cmd' field: {item}")

        # Write CSV (one command per row, no header)
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for cmd in unique_commands:
                writer.writerow([cmd])

        print(f"Conversion complete: processed {len(data)} records")
        print(f"Removed {duplicate_count} duplicate commands")
        print(f"Kept {len(unique_commands)} unique commands")
        print(f"Saved to {csv_file_path}")

    except FileNotFoundError:
        print(f"Error: file not found {json_file_path}")
    except json.JSONDecodeError:
        print(f"Error: {json_file_path} is not valid JSON")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")


if __name__ == "__main__":
    input_json = "/Users/florence/Desktop/commands.json"
    output_csv = "commands.csv"
    json_to_cmd_csv(input_json, output_csv)
