import json
from dataset import Dataset
import nltk

def formality_score(nounFreq, adjectiveFreq, prepositionFreq, articleFreq, pronounFreq, verbFreq, adverbFreq, interjectionFreq):
    return (nounFreq + adjectiveFreq + prepositionFreq + articleFreq - pronounFreq - verbFreq - adverbFreq - interjectionFreq + 100) / 2

def annotate_formality(folder):
    print("Now annotating all paragraphs.")
    dataset = Dataset(folder)

    annotations = []
    progress = 0

    for element in dataset.get_elements():
        progress += 1
        print(f"{progress}/{len(dataset.get_elements())}")
        annotation = {}

        annotation["id"] = element.element_id

        for (tag, el) in [("post_text", element.post_text), ("target_paragraphs", " ".join(element.target_paragraphs))]:
            text = " ".join(el)
            tokenize = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(tokenize)

            nounFreq = len(list(filter(lambda x: x[1].startswith("NN"), pos_tags)))
            adjectiveFreq = len(list(filter(lambda x: x[1].startswith("JJ"), pos_tags)))
            prepositionFreq = len(list(filter(lambda x: x[1].startswith("IN"), pos_tags)))
            artiqleFreq = len(pos_tags)
            pronounFreq = len(list(filter(lambda x: x[1].startswith("PR"), pos_tags)))
            verbFreq = len(list(filter(lambda x: x[1].startswith("VB"), pos_tags)))
            adverbFreq = len(list(filter(lambda x: x[1].startswith("RB"), pos_tags)))
            interjectionFreq = len(list(filter(lambda x: x[1].startswith("UH"), pos_tags)))

            # Retrieve annotation information
            annotation[f"{tag}_formality"] = formality_score(nounFreq, adjectiveFreq, prepositionFreq, artiqleFreq, pronounFreq, verbFreq, adverbFreq, interjectionFreq)
            annotations.append(annotation)

    f = open(folder + "/formality_annotations.jsonl", "w+")

    for annotation in annotations:
        f.write(json.dumps(annotation))
        f.write("\n")

    f.close()

annotate_formality("datasets/small_training")
annotate_formality("datasets/big_training")