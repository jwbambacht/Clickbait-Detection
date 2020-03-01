# Clickbait-Detection

## Requirements
| Name        | Version | Install                   | Description                      |
|-------------|---------|---------------------------|----------------------------------|
| pytesseract | 0.3.2   | `pip install pytesseract` | 'Reads' text embedded in images. This also requires tesseract to be installed on your system and in your PATH. This library is only used to annotate the images in `annotate_images.py`, results are already saved (see: `media_annotations.jsonl`).|
| pyenchant | 2.0.0  | `pip install pyenchant`| Used to identify the formality of words. |
| NumPy | 1.18.1 | `pip install numpy` | Used for scientific computing.|
## Data

| Filename | Location | # posts |  % clickbait | Download link |
|----------|----------|---------------|----|----|
|  clickbait17-train-170331.zip        |  [/datasets/big_training](/datasets/big_training)        |      19538         | 24.4% | [link](http://www.uni-weimar.de/medien/webis/corpora/corpus-webis-clickbait-17/clickbait17-train-170331.zip)|
|   clickbait17-train-170630.zip       |   [/datasets/small_training](/datasets/small_training)       |       2495        | 30.1% | [link](http://www.uni-weimar.de/medien/webis/corpora/corpus-webis-clickbait-17/clickbait17-train-170630.zip)|


**Note:** the `/media` folders have been removed from the repo due to the size. Download them yourself and copy to the right location.

Data can be loaded as follows:
```python
# Provide the directory path containing instances.
# Needs to include: instances.jsonl, truth.jsonl and /media folder.
small_dataset = Dataset("datasets/small_training")
```
To get an overview of the dataset: `small_dataset.print_summary()`, which outputs:
```bash
--- Dataset Summary --
Directory: datasets/small_training
Amount of elements: 2459
Percentage clickbait: 30.98820658804392
Percentage non-clickbait: 69.01179341195608
----------------------
```

### Features
| # | Name          | Description                            |
|-------|---------------|---------------------------------|
| 1 | Image availability | Checks if post has an image available. |
| 2 | Image text availability | Checks if a post image has text.   |
| 3-9 | Length of article content | Counts amount of characters in the `post_title`, `media_text`, `target_title`, `target_description`, `target_keywords`, `target_captions` and `target_paragraphs` fields.| 

To get the features, use `[small|big]_dataset.get_features()`:
```bash
 [  1.   1.  98. ... 129.   3.  36.]
 [  0.   0.  89. ...   0.  16.   3.]
 ...
 [  1.   0.  65. ...   0.  34.   0.]
 [  0.   0.  88. ... 131.  15.   5.]
 [  1.   0.  57. ...   0.  11.   0.]]
```
It returns an `instances * features` matrix. Its loaded from a file by default, to regenerate this matrix use the `overwrite=True` flag.
To retrieve the corresponding (target) labels, use `[small|big]_dataset.get_target_labels()`:
```bash
[0 0 1 ... 0 0 0]
```

## PART 1:
- [x] Read data
- [x] Feature extraction
- [ ] Come up with new features (based on NLP theory)

## PART 2:
- [ ] Train classifier(s) (which classifier is best and why)
- [ ] Explain potential overfitting
- [ ] Explain potential bias
- [ ] How do we deal with bias and overfitting?
- [ ] Cross-validation why (not)?
## PART 3:
- [ ] Evaluate results
- [ ] Which metrics are (not) useful?
- [ ] Which conclusions can we (not) make?
- [ ] Maybe give some further recommendations on how to improve? (Like more data, different ML techniques etc.)
