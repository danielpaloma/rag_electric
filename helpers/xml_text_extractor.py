import os
import xml.etree.ElementTree as ET

def extract_plain_text_from_sec_file(file_path):
    """
    Extract all plain text from a .SEC (XML) file, ignoring all tags and attributes.
    Returns a single string with the concatenated text content.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        # Recursively join all text and tail content
        texts = []
        for elem in root.iter():
            if elem.text:
                texts.append(elem.text)
            if elem.tail:
                texts.append(elem.tail)
        # Join and clean up whitespace
        return '\n'.join(t.strip() for t in texts if t and t.strip())
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""

def extract_and_save_sec_as_txt(sec_file_path, target_folder):
    """
    Extract plain text from a .SEC file and save it as a .txt file in the target folder.
    The output .txt file will have the same base filename as the .SEC file.
    """
    plain_text = extract_plain_text_from_sec_file(sec_file_path)
    if not plain_text:
        print(f"No text extracted from {sec_file_path}, skipping save.")
        return None
    os.makedirs(target_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(sec_file_path))[0]
    txt_file_path = os.path.join(target_folder, base_name + ".txt")
    with open(txt_file_path, "w", encoding="utf-8") as f:
        f.write(plain_text)
    print(f"Saved plain text to {txt_file_path}")
    return txt_file_path 