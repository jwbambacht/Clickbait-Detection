import json

# Stores a set of instances and its label (clickbait or not).
class Dataset:
    def __init__(self, directory):
        self.directory = directory
        self.instances_file = directory + "/instances.jsonl"
        self.truth_file = directory + "/truth.jsonl"
        self.elements = {}

        self.__load_instances()

    # Loads the element instances and its truths.
    def __load_instances(self):
        for e in self.read_data(self.instances_file):
            element = self.parse_element(e)
            self.elements[element.element_id] = element

        for t in self.read_data(self.truth_file):
            truth = self.parse_truth(t)
            self.elements[truth.element_id].set_truth(truth)

    # Reads jsonl file.
    @staticmethod
    def read_data(file):
        with open(file) as f:
            for line in f:
                yield json.loads(line)

    # Parses dict entry into Element class.
    @staticmethod
    def parse_element(e):
        return Element(
            e["id"],
            e["postTimestamp"],
            e["postText"],
            e["postMedia"],
            e["targetTitle"],
            e["targetDescription"],
            e["targetKeywords"],
            e["targetParagraphs"],
            e["targetCaptions"],
        )

    # Parses dict entry into Truth class.
    @staticmethod
    def parse_truth(t):
        return Truth(
            t["id"],
            t["truthJudgments"],
            t["truthMean"],
            t["truthMedian"],
            t["truthMode"],
            t["truthClass"],
        )

    # Returns the amount of elements in the dataset.
    def amount_of_elements(self):
        return len(self.elements)

    # Returns the list of elements in the dataset.
    def get_elements(self):
        return list(self.elements.values())

    # Returns the amount of click bait elements in the dataset.
    def amount_of_clickbait(self):
        return sum(map(lambda x: x.get_truth().is_clickbait(), self.elements.values()))

    # Prins a summary of this dataset.
    def print_summary(self):
        print("--- Dataset Summary --")
        print(f"Directory: {self.directory}")
        print(f"Amount of elements: {self.amount_of_elements()}")
        clickbait_perc = (self.amount_of_clickbait() / self.amount_of_elements()) * 100
        print(f"Percentage clickbait: {clickbait_perc}")
        print(f"Percentage non-clickbait: {100 - clickbait_perc}")
        print("----------------------")

# Stores an instance/sample with all its metadata and correct label.
class Element:
    def __init__(
        self,
        element_id,
        timestamp,
        post_text,
        post_media,
        target_title,
        target_description,
        target_keywords,
        target_paragraphs,
        target_captions,
    ):
        self.element_id = element_id
        self.timestamp = timestamp
        self.post_text = post_text
        self.post_media = post_media
        self.target_title = target_title
        self.target_description = target_description
        self.target_keywords = target_keywords
        self.target_paragraphs = target_paragraphs
        self.target_captions = target_captions
        self.__truth = None

    # Set the truth of this element.
    def set_truth(self, truth):
        self.__truth = truth

    # Get the truth of this element.
    def get_truth(self):
        if self.__truth is None:
            raise ValueError("Truth is not assigned (yet).")

        return self.__truth

    ## FEATURE EXTRACTION

    # Feature 1
    def __has_image(self):
        return int(len(self.post_media) >= 1)

    # Print this element.
    def pretty_print(self, verbose=False):
        print("-- Element --")
        print(f"id: {self.element_id}")
        print(f"title: {self.target_title}")
        print(f"timestamp: {self.timestamp}")
        print(f"is_clickbait: {bool(self.get_truth().is_clickbait())}")

        if verbose:
            print(f"post_text: {self.post_text}")
            print(f"post_media: {self.post_media}")
            print(f"target_description: {self.target_description}")
            print(f"target_keywords: {self.target_keywords}")
            print(f"target_paragraphs: {self.target_paragraphs}")
            print(f"target_captions: {self.target_captions}")

        print("-------------")


# Stores the labels of the samples/instances/elements (also with a certain probability based on crowd sourcing).
class Truth:
    def __init__(
        self,
        element_id,
        truth_judgments,
        truth_mean,
        truth_median,
        truth_mode,
        truth_class,
    ):
        self.element_id = element_id
        self.truth_judgments = truth_judgments
        self.truth_mean = truth_mean
        self.truth_median = truth_median
        self.truth_mode = truth_mode
        self.truth_class = truth_class

    # Returns if the post is labelled clickbait.
    def is_clickbait(self):
        return int(self.truth_class == "clickbait")
