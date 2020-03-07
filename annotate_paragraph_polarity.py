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

        vs = analyzer.polarity_scores(" ".join(element.target_paragraphs))

        # Retrieve annotation information
        annotation["id"] = element.element_id
        annotation["pos"] = vs["pos"]
        annotation["neg"] = vs["neg"]
        annotation["neu"] = vs["neu"]
        annotation["compound"] = vs["compound"]

        annotations.append(annotation)

    f = open(folder + "/polarity_annotations.jsonl", "w+")

    for annotation in annotations:
        f.write(json.dumps(annotation))
        f.write("\n")

    f.close()


annotate_polarity("datasets/big_training")