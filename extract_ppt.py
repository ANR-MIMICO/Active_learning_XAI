import zipfile
import xml.etree.ElementTree as ET
import os

def extract_text_from_pptx(path):
    text_runs = []
    try:
        with zipfile.ZipFile(path, 'r') as z:
            for filename in z.namelist():
                if filename.startswith('ppt/slides/slide') and filename.endswith('.xml'):
                    content = z.read(filename)
                    tree = ET.fromstring(content)
                    for elem in tree.iter():
                        if elem.tag.endswith('}t'):
                            text_runs.append(elem.text)
    except Exception as e:
        print(f"Error: {e}")
    return text_runs

texts = extract_text_from_pptx(r"C:\Users\psaves\Desktop\Active_learning_XAI\v0.pptx")
print("\n".join(t for t in texts if t))
