import json
import os
import re
from itertools import combinations
import enchant
from operator import add
import numpy as np

# Stores a set of instances and its label (clickbait or not).
class Dataset:
    def __init__(self, directory):
        self.directory = directory
        self.instances_file = directory + "/instances.jsonl"
        self.truth_file = directory + "/truth.jsonl"
        self.media_annotations_file = directory + "/media_annotations.jsonl"
        self.polarity_annotations_file = directory + "/polarity_annotations.jsonl"
        self.elements = {}
        self.media_annotations = {}
        self.word_dict = {}
        self.polarity_annotations = {}

        self.__load_instances()
        self.__build_dictionary()

    # Loads the element instances and its truths.
    def __load_instances(self):
        for e in self.read_data(self.instances_file):
            element = self.parse_element(e)
            self.elements[element.element_id] = element

        for t in self.read_data(self.truth_file):
            truth = self.parse_truth(t)
            self.elements[truth.element_id].set_truth(truth)

        # If media annotations don't exist, ignore it.
        if os.path.exists(self.media_annotations_file):
            # Add media annotations.
            for m in self.read_data(self.media_annotations_file):
                self.media_annotations[m["id"]] = m

            # Make sure elements retrieve their own annotations.
            for el in self.get_elements():
                if len(el.post_media) > 0:
                    el.set_image_annotations(self.media_annotations)

        # If polarity annotations don't exist, ignore it.
        if os.path.exists(self.polarity_annotations_file):
            # Add polarity annotations.
            for p in self.read_data(self.polarity_annotations_file):
                self.polarity_annotations[p["id"]] = p

            # Make sure elements retrieve their own annotations.
            for el in self.get_elements():
                el.set_polarity_annotation(self.polarity_annotations[el.element_id])

    def __build_dictionary(self):
        all_words = []
        for el in self.get_elements():
            all_words += el.get_all_words()

        # Remove duplicates
        all_words = list(set(all_words))

        # Add words to dictionary with True or False. True if valid, otherwise False.
        for w in all_words:
            d = enchant.Dict("en_US")
            self.word_dict[w] = d.check(w)

        for el in self.get_elements():
            el.word_dict = self.word_dict

    # Reads jsonl file.
    @staticmethod
    def read_data(file):
        with open(file, encoding="utf8") as f:
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

    # Get features of the whole dataset.
    def get_features(self, overwrite=False, filename="features.npy"):
        path = self.directory + f"/{filename}"
        if os.path.exists(path) and not overwrite:
            print(
                "Loading from file! If you want to overwrite the features, use overwrite = True."
            )
            return np.load(path)

        # Get all features.
        arrays = [el.get_features() for el in self.get_elements()]
        np_array = np.stack(arrays, axis=0)

        # Save to file.
        np.save(path, np_array)

        return np.stack(arrays, axis=0)

    # Get all target labels.
    def get_target_labels(self):
        targets = [el.get_truth().is_clickbait() for el in self.get_elements()]
        return np.array(targets)

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
        self.word_dict = {}
        self.polarity_annotation = {}
        self.__truth = None
        self.__media_text_present = [False] * len(self.post_media)
        self.__media_text = [""] * len(self.post_media)

    # Set the truth of this element.
    def set_truth(self, truth):
        self.__truth = truth

    # Get the truth of this element.
    def get_truth(self):
        if self.__truth is None:
            raise ValueError("Truth is not assigned (yet).")

        return self.__truth

    # Sets polarity annotation.
    def set_polarity_annotation(self, annotation):
        self.polarity_annotation = annotation

    # Set an image annotation.
    def set_image_annotations(self, annotations):
        for m in self.post_media:  # All media.
            index = self.post_media.index(m)
            annotation = annotations[m.replace("media/", "")]

            # Update annotation.
            self.__media_text_present[index] = annotation["has_text"]
            self.__media_text[index] = annotation["text"]

    # START - FEATURE EXTRACTION
    # Feature 1
    def __has_image(self):
        return int(len(self.post_media) >= 1)

    # Feature 2
    def __has_image_text(self):
        if not bool(self.__has_image()):
            return 0
        else:
            return int(True in self.__media_text_present)

    # Feature 3
    def __post_title_len(self):
        return self.__zero_check(self.__sum_list(self.post_text, len))

    # Feature 4
    def __text_in_media_len(self):
        return self.__zero_check(self.__sum_list(self.__media_text, len))

    # Feature 5
    def __target_title_len(self):
        return self.__zero_check(len(self.target_title))

    # Feature 6
    def __target_description_len(self):
        return self.__zero_check(len(self.target_description))

    # Feature 7
    def __target_keywords_len(self):
        return self.__zero_check(self.__sum_list(self.target_keywords, len))

    # Feature 8
    def __target_captions_len(self):
        return self.__zero_check(self.__sum_list(self.target_captions, len))

    # Feature 9
    def __target_paragraphs_len(self):
        return self.__zero_check(self.__sum_list(self.target_paragraphs, len))

    # Feature 10 - 30
    def __diff_num_of_characters(self):
        chars = self.__list_chars()

        diff = []
        for (charA, charB) in combinations(chars, 2):
            diff.append(abs(charA() - charB()))

        return diff

    # Feature 31 - 51
    def __num_of_characters_ratio(self):
        chars = self.__list_chars()

        ratio = []
        for (charA, charB) in combinations(chars, 2):
            if charA() is -1 or charB() is -1:
                ratio.append(-1)
            else:
                ratio.append(abs(charA() / charB()))

        return ratio

    # Feature 52
    def __post_title_word_count(self):
        return self.__zero_check(self.__sum_list(self.post_text, self.__count_words))

    # Feature 53
    def __text_in_media_word_count(self):
        return self.__zero_check(self.__sum_list(self.__media_text, self.__count_words))

    # Feature 54
    def __target_title_word_count(self):
        return self.__zero_check(self.__count_words(self.target_title))

    # Feature 55
    def __target_description_word_count(self):
        return self.__zero_check(self.__count_words(self.target_description))

    # Feature 56
    def __target_keywords_word_count(self):
        return self.__zero_check(
            self.__sum_list(self.target_keywords, self.__count_words)
        )

    # Feature 57
    def __target_captions_word_count(self):
        return self.__zero_check(
            self.__sum_list(self.target_captions, self.__count_words)
        )

    # Feature 58
    def __target_paragraphs_word_count(self):
        return self.__zero_check(
            self.__sum_list(self.target_paragraphs, self.__count_words)
        )

    # Feature 59 - 79
    def __diff_num_of_words(self):
        words = self.__list_words()

        diff = []
        for (wordA, wordB) in combinations(words, 2):
            diff.append(abs(wordA() - wordB()))

        return diff

    # Feature 80 - 100
    def __num_of_words_ratio(self):
        words = self.__list_words()

        ratio = []
        for (wordA, wordB) in combinations(words, 2):
            if wordA() is -1 or wordB() is -1:
                ratio.append(-1)
            else:
                ratio.append(abs(wordA() / wordB()))

        return ratio

    # Feature 101 - 106
    def __num_common_keywords_article(self):
        keywords = set(
            [
                el.strip()
                for el in re.findall(r"\w+", " ".join(self.target_keywords.split(",")))
            ]
        )
        words = [
            self.post_text,
            self.__media_text,
            self.target_title,
            self.target_description,
            self.target_captions,
            self.target_paragraphs,
        ]

        overlap = []

        for w in words:
            if isinstance(w, list):
                all_words = [el.strip() for el in re.findall(r"\w+", " ".join(w))]
            else:
                all_words = [el.strip() for el in re.findall(r"\w+", w)]

            overlap.append(len(keywords.difference(set(all_words))))

        return overlap

    # Feature 107 to 113
    def __num_of_formal_words(self):
        words = [
            self.post_text,
            self.__media_text,
            self.target_title,
            self.target_description,
            self.target_captions,
            self.target_keywords,
            self.target_paragraphs,
        ]

        num_words = []
        for w in words:
            if isinstance(w, list):
                distinct_words = set(re.findall(r"\w+", " ".join(w)))
            else:
                distinct_words = set(re.findall(r"\w+", w))

            num_words.append(sum([self.word_dict[word] for word in distinct_words]))

        return num_words

    # Feature 114 to 120
    def __num_of_informal_words(self):
        words = [
            self.post_text,
            self.__media_text,
            self.target_title,
            self.target_description,
            self.target_captions,
            self.target_keywords,
            self.target_paragraphs,
        ]

        num_words = []
        for w in words:
            if isinstance(w, list):
                distinct_words = set(re.findall(r"\w+", " ".join(w)))
            else:
                distinct_words = set(re.findall(r"\w+", w))

            num_words.append(sum([not self.word_dict[word] for word in distinct_words]))

        return num_words

    # Feature 121 to 127
    def __ratio_formal_words(self):
        formal_words = self.__num_of_formal_words()
        informal_words = self.__num_of_informal_words()
        total_words = map(add, formal_words, informal_words)

        formal_ratio = []
        for formal, total in zip(formal_words, total_words):
            if total is 0:
                formal_ratio.append(0)
                continue
            formal_ratio.append(formal / total)

        return formal_ratio

    # Feature 128 to 134
    def __ratio_informal_words(self):
        formal_words = self.__num_of_formal_words()
        informal_words = self.__num_of_informal_words()
        total_words = map(add, formal_words, informal_words)

        informal_ratio = []
        for informal, total in zip(informal_words, total_words):
            if total is 0:
                informal_ratio.append(0)
                continue
            informal_ratio.append(informal / total)

        return informal_ratio

    # Feature 135
    def __num_of_at(self):
        return self.__count_element_in_post("@")

    # Feature 136
    def __num_of_hashtags(self):
        return self.__count_element_in_post("#")

    # Feature 137
    def __num_of_retweets(self):
        return self.__count_element_in_post("RT") + self.__count_element_in_post(
            "retweet"
        )

    # Feature 138
    def __num_of_additional_symbols(self):
        return (
            self.__count_element_in_post("\?")
            + self.__count_element_in_post("\,")
            + self.__count_element_in_post("\:")
            + self.__count_element_in_post("\.\.\.")
        )

    # Feature 139
    def __num_of_keywords(self):
        return len(self.target_keywords)

    # Feature 140
    def __num_of_paragraphs(self):
        return len(self.target_paragraphs)

    # Feature 141
    def __num_of_captions(self):
        return len(self.target_captions)

    # Feature 142
    def __positive_sentiment(self):
        return self.polarity_annotation["pos"]

    # Feature 143
    def __negative_sentiment(self):
        return self.polarity_annotation["neg"]

    # Feature 144
    def __neutral_sentiment(self):
        return self.polarity_annotation["neu"]

    # END - FEATURE EXTRACTION

    # Get features as numpy array.
    def get_features(self):
        return np.hstack(
            [
                self.__has_image(),
                self.__has_image_text(),
                self.__post_title_len(),
                self.__text_in_media_len(),
                self.__target_title_len(),
                self.__target_description_len(),
                self.__target_keywords_len(),
                self.__target_captions_len(),
                self.__target_paragraphs_len(),
                self.__diff_num_of_characters(),
                self.__num_of_characters_ratio(),
                self.__post_title_word_count(),
                self.__text_in_media_word_count(),
                self.__target_title_word_count(),
                self.__target_description_word_count(),
                self.__target_keywords_word_count(),
                self.__target_captions_word_count(),
                self.__target_paragraphs_word_count(),
                self.__diff_num_of_words(),
                self.__num_of_words_ratio(),
                self.__num_common_keywords_article(),
                self.__num_of_formal_words(),
                self.__num_of_informal_words(),
                self.__ratio_formal_words(),
                self.__ratio_informal_words(),
                self.__num_of_at(),
                self.__num_of_hashtags(),
                self.__num_of_retweets(),
                self.__num_of_additional_symbols(),
                self.__num_of_keywords(),
                self.__num_of_paragraphs(),
                self.__num_of_captions(),
                self.__positive_sentiment(),
                self.__negative_sentiment(),
                self.__neutral_sentiment()
            ]
        )

    # Returns -1 if value is equal to 0.
    def __zero_check(self, value):
        return -1 if value is 0 else value

    # Returns the sum of the length of the characters in a list.
    def __sum_list(self, list, fun):
        return sum(map(lambda x: fun(x), list))

    # Count amount of words in string.
    def __count_words(self, str):
        return len(re.findall(r"\w+", str))

    # Counts an element occurence in a post.
    def __count_element_in_post(self, element):
        words = [
            self.post_text,
            self.__media_text,
            self.target_title,
            self.target_description,
            self.target_captions,
            self.target_keywords,
            self.target_paragraphs,
        ]

        count = 0
        for w in words:
            if isinstance(w, list):
                w = " ".join(w)

            count += len(re.findall(rf"{element}", w))

        return count

    # Retrieves all words from the article.
    def get_all_words(self):
        words = [
            self.post_text,
            self.__media_text,
            self.target_title,
            self.target_description,
            self.target_captions,
            self.target_keywords,
            self.target_paragraphs,
        ]

        all_words = []

        for w in words:
            if isinstance(w, list):
                all_words += [el.strip() for el in re.findall(r"\w+", " ".join(w))]
            else:
                all_words += [el.strip() for el in re.findall(r"\w+", w)]

        # Remove duplicates.
        return list(set(all_words))

    # Function handles to all character lengths.
    def __list_chars(self):
        return [
            self.__post_title_len,
            self.__text_in_media_len,
            self.__target_title_len,
            self.__target_description_len,
            self.__target_keywords_len,
            self.__target_captions_len,
            self.__target_paragraphs_len,
        ]

    def __list_words(self):
        return [
            self.__post_title_word_count,
            self.__text_in_media_word_count,
            self.__target_title_word_count,
            self.__target_description_word_count,
            self.__target_keywords_word_count,
            self.__target_captions_word_count,
            self.__target_paragraphs_word_count,
        ]

    # Print this element.
    def pretty_print(self, verbose=False):
        print("-- Element --")
        print(f"id: {self.element_id}")
        print(f"post_title: {self.post_text}")
        print(f"timestamp: {self.timestamp}")
        print(f"is_clickbait: {bool(self.get_truth().is_clickbait())}")

        if verbose:
            print(f"post_media: {self.post_media}")
            print(f"target_title: {self.target_title}")
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
