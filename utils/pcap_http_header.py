"""
pcap_http_headers.py

This utility parses a PCAP file with Scapy, reconstructs TCP streams,
and extracts HTTP request lines and headers into a single-column CSV.

- Input:  A `.pcap` file containing captured network traffic.
- Output: A CSV file where each row is either:
    * An HTTP request line (e.g., "GET /path HTTP/1.1")
    * Or an HTTP request header (e.g., "User-Agent: ...")

Features:
- Reassembles TCP streams by 4-tuple (src, sport, dst, dport).
- Splits multiple HTTP requests within a single stream.
- Handles multi-line headers (folded lines).
- Skips excluded headers (by default "Host" and "Server").
- Outputs a clean CSV with one value per row.

Example:
    python pcap_http_headers.py

Author: Florence
"""

from scapy.all import *
from scapy.layers.inet import TCP, IP
import sys
import re
import csv
from collections import defaultdict

# Excluded HTTP header fields (edit to customize)
EXCLUDED_HEADERS = {"Host", "Server"}  # Set to {} to keep all headers


def extract_http_headers(pcap_file, output_file):
    print(f"[*] Parsing {pcap_file}...")

    try:
        # Reconstruct TCP streams: use 4-tuple as key
        tcp_streams = defaultdict(bytes)
        packets = rdpcap(pcap_file)

        print("[*] Reassembling TCP streams...")
        for pkt in packets:
            if IP in pkt and TCP in pkt and pkt[TCP].payload:
                stream_id = (pkt[IP].src, pkt[TCP].sport, pkt[IP].dst, pkt[TCP].dport)
                tcp_streams[stream_id] += bytes(pkt[TCP].payload)

        print(f"[*] Reassembly complete, found {len(tcp_streams)} TCP streams")

        total_entries = 0

        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Analyze each TCP stream
            for stream_id, stream_data in tcp_streams.items():
                try:
                    http_requests = split_http_requests(stream_data)

                    for req in http_requests:
                        try:
                            req_str = req.decode('utf-8', errors='replace')
                        except:
                            req_str = req.decode('latin-1', errors='replace')

                        lines = req_str.split('\r\n')
                        if not lines:
                            continue

                        # Write request line
                        request_line = lines[0].strip()
                        if request_line:
                            writer.writerow([request_line])
                            total_entries += 1

                        # Write request headers
                        in_headers = True
                        current_header = None
                        for line in lines[1:]:
                            if not line.strip():  # End of headers
                                if current_header:
                                    header_str = f"{current_header['key']}: {current_header['value']}"
                                    if current_header['key'] not in EXCLUDED_HEADERS:
                                        writer.writerow([header_str])
                                        total_entries += 1
                                    current_header = None
                                break

                            # Continuation of previous header
                            if re.match(r'^\s+', line):
                                if current_header:
                                    current_header['value'] += ' ' + line.strip()
                                continue

                            # Normal header
                            if ':' in line:
                                if current_header:
                                    header_str = f"{current_header['key']}: {current_header['value']}"
                                    if current_header['key'] not in EXCLUDED_HEADERS:
                                        writer.writerow([header_str])
                                        total_entries += 1
                                key, value = line.split(':', 1)
                                current_header = {'key': key.strip(), 'value': value.strip()}

                except Exception as e:
                    print(f"[!] Error processing stream {stream_id}: {str(e)}")
                    continue

        print(f"[+] Done. Extracted {total_entries} request lines + headers")
        print(f"[+] Output saved to {output_file}")

    except Exception as e:
        print(f"[-] Fatal error: {str(e)}")
        sys.exit(1)


def split_http_requests(stream_data):
    """Split multiple HTTP requests within a TCP stream."""
    http_requests = []
    request_starts = []

    methods = [b'GET ', b'POST ', b'PUT ', b'DELETE ', b'HEAD ', b'OPTIONS ', b'CONNECT ']
    for method in methods:
        pos = 0
        while True:
            idx = stream_data.find(method, pos)
            if idx == -1:
                break
            if idx == 0 or (idx >= 2 and stream_data[idx - 2:idx] == b'\r\n'):
                request_starts.append(idx)
            pos = idx + len(method)

    request_starts.sort()
    for i in range(len(request_starts)):
        start = request_starts[i]
        end = request_starts[i + 1] if (i + 1 < len(request_starts)) else len(stream_data)
        http_requests.append(stream_data[start:end])

    return http_requests


if __name__ == "__main__":
    extract_http_headers(
        pcap_file="/Users/florence/Desktop/bigFlows.pcap",
        output_file="http_headers.csv"
    )
