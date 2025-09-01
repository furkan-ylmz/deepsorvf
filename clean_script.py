#!/usr/bin/env python3
import os
import re
import glob

def clean_chinese_content():
    # Pattern for Chinese characters
    todo_pattern = re.compile(r'^\s*(TODO:|FIXME:|XXX:)', re.IGNORECASE)
    
    # Find all Python files
    py_files = glob.glob("**/*.py", recursive=True)
    
    for file_path in py_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            cleaned_lines = []
            skip_multiline = False
            
            for line in lines:
                # Skip Chinese content in comments and docstrings
                if chinese_pattern.search(line):
                    # If line has Chinese, skip it
                    continue
                
                # Skip TODO/FIXME/XXX comments
                if todo_pattern.match(line.strip()):
                    continue
                
                # Keep the line
                cleaned_lines.append(line)
            
            # Write back cleaned content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(cleaned_lines)
                
            print(f"Cleaned: {file_path}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    clean_chinese_content()
