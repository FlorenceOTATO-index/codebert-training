import os
import base64
import argparse

def process_directory(root_dir, output_file, skip_extensions, skip_filenames):
    """
    Recursively process the directory, read file contents, and generate Base64 strings
    """
    # Initialize counters
    total_files = 0
    skipped_by_filter = 0
    skipped_by_encoding = 0
    processed_files = 0
    error_files = 0
    
    with open(output_file, 'w') as out_f:
        for root, dirs, files in os.walk(root_dir):
            # Remove hidden directories (starting with .) to avoid traversal
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            total_files += len(files)
            
            for filename in files:
                file_path = os.path.join(root, filename)
                
                # Check if the file should be skipped (including hidden files)
                if should_skip_file(filename, skip_extensions, skip_filenames):
                    print(f"Skipping (filter): {file_path}")
                    skipped_by_filter += 1
                    continue
                
                try:
                    # Read file content
                    with open(file_path, 'rb') as in_f:
                        content = in_f.read()
                    
                    # Check if file encoding is valid UTF-8
                    if not is_valid_utf8(content):
                        print(f"Skipping (invalid encoding): {file_path}")
                        skipped_by_encoding += 1
                        continue
                    
                    # Generate Base64 string and append ",1"
                    base64_str = base64.b64encode(content).decode('utf-8')
                    output_line = f"{base64_str},1\n"
                    
                    # Write to output file
                    out_f.write(output_line)
                    print(f"Processed: {file_path}")
                    processed_files += 1
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    error_files += 1
    
    # Return statistics
    return {
        "total_files": total_files,
        "skipped_by_filter": skipped_by_filter,
        "skipped_by_encoding": skipped_by_encoding,
        "processed_files": processed_files,
        "error_files": error_files
    }

def should_skip_file(filename, skip_extensions, skip_filenames):
    """
    Check whether the file should be skipped
    """
    # 1. Skip hidden files (starting with .)
    if filename.startswith('.'):
        return True
    
    # 2. Skip specific filenames
    if filename in skip_filenames:
        return True
    
    # 3. Skip files without extensions
    if '.' not in filename:
        return True
    
    # 4. Skip by extension
    ext = filename.split('.')[-1].lower()
    # if ext in [e.lower() for e in skip_extensions]:
    if ext != 'php':
        return True
    
    return False

def is_valid_utf8(content):
    """
    Check if the content is valid UTF-8 encoding
    """
    # Empty files are considered valid
    if not content:
        return True
    
    try:
        # Try decoding as UTF-8
        content.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False

def main():
    # Configure arguments
    parser = argparse.ArgumentParser(description='Recursively process files and generate Base64 encoding')
    parser.add_argument('directory', help='Root directory to process')
    parser.add_argument('-o', '--output', default='output.txt', help='Output filename')
    args = parser.parse_args()
    
    # Configure extensions to skip (modifiable)
    SKIP_EXTENSIONS = [
        'md', 'MD', 'yml', 'py', 'zip', 'rar', '7z', 'me', 'xml', 'js', 'ini',
        'PNG', 'png', 'jpg', 'gif', 'war', 'ear', 'jar', 'json', 'scss',
        'doc', 'css', 'txt', 'pdf', 'mhtml', 'gz', 'tar.gz', 'tar', 'cs', 'htm', 'html'
    ]

    # Configure specific filenames to skip (modifiable)
    SKIP_FILENAMES = [
        'README', '.DS_Store', 'Thumbs.db', 'desktop.ini'
    ]
    
    print(f"Starting processing from: {args.directory}")
    print(f"Output will be saved to: {args.output}")
    print(f"Skipping extensions: {SKIP_EXTENSIONS}")
    print(f"Skipping filenames: {SKIP_FILENAMES}")
    print("Skipping hidden files/directories (starting with '.')")
    
    # Process directory and get statistics
    stats = process_directory(
        root_dir=args.directory,
        output_file=args.output,
        skip_extensions=SKIP_EXTENSIONS,
        skip_filenames=SKIP_FILENAMES
    )

    # Print summary
    print("\n===== Processing Summary =====")
    print(f"Total files found:         {stats['total_files']}")
    print(f"Files skipped by filter:   {stats['skipped_by_filter']}")
    print(f"Files skipped by encoding: {stats['skipped_by_encoding']}")
    print(f"Files processed:           {stats['processed_files']}")
    print(f"Files with errors:         {stats['error_files']}")
    print(f"Output saved to:           {args.output}")


if __name__ == "__main__":
    main()
