import os
from helpers.xml_text_extractor import extract_and_save_sec_as_txt
from pathlib import Path

# Define source and target directories relative to the project root
SOURCE_DIR = Path(__file__).resolve().parent.parent / "knowledge-base" / "raw_data" / "UFGS_DIVISION_26"
TARGET_DIR = Path(__file__).resolve().parent.parent / "knowledge-base" / "clean_data" / "UFGS_div26_txt"

def main():
    if not SOURCE_DIR.exists():
        print(f"ERROR: Source directory does not exist: {SOURCE_DIR}")
        return

    sec_files = list(SOURCE_DIR.glob("*.SEC"))
    print(f"Found {len(sec_files)} .SEC files in {SOURCE_DIR}")

    if not sec_files:
        print("No .SEC files found. Exiting.")
        return

    for sec_file in sec_files:
        print(f"Processing: {sec_file.name}")
        extract_and_save_sec_as_txt(sec_file, TARGET_DIR)

if __name__ == "__main__":
    main() 