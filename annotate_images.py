import glob
import json

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract


# If you don't have tesseract executable in your PATH, include the following:
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'


def annotate_media(folder):
    print("Now annotating all images.")
    media_folder = folder + "/media"

    annotations = []
    files = glob.glob(f"{media_folder}/*")
    progress = 0
    for file in files:
        progress += 1
        print(f"{progress}/{len(files)}")
        annotation = {}

        # Retrieve annotation information
        image_text = pytesseract.image_to_string(file)
        annotation["id"] = file.replace(media_folder + "/", "")
        annotation["has_text"] = int(len(image_text) > 0)
        annotation["text"] = image_text

        annotations.append(annotation)

    f = open(folder + "/media_annotations.jsonl", "w+")

    for annotation in annotations:
        f.write(json.dumps(annotation))
        f.write("\n")

    f.close()


annotate_media("datasets/small_training")
annotate_media("datasets/big_training")
