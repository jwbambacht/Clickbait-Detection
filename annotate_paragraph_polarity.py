import glob
import json
from dataset import Dataset
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

def annotate_polarity(folder):
    print("Now annotating all paragraphs.")
    dataset = Dataset(folder)

    annotations = []
    progress = 0

    for element in dataset.get_elements():
        progress += 1
        print(f"{progress}/{len(dataset.get_elements())}")
        annotation = {}

        annotation["id"] = element.element_id

        for (tag, el) in [("post_text", " ".join(element.post_text)), ("target_title", element.target_title), ("target_paragraphs", " ".join(element.target_paragraphs))]:
            vs = analyzer.polarity_scores(el)

            # Retrieve annotation information
            annotation[f"{tag}_compound"] = vs["compound"]
            annotations.append(annotation)

    f = open(folder + "/polarity_annotations.jsonl", "w+")

    for annotation in annotations:
        f.write(json.dumps(annotation))
        f.write("\n")

    f.close()


annotate_polarity("datasets/small_training")
annotate_polarity("datasets/big_training")