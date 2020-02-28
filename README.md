# Clickbait-Detection

## Requirements

## Data

| Filename | Location | # posts |  Percentage clickbait | Download link |
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

## PART 1:
* Read data
* Feature extraction
* Come up with new features (based on NLP theory)

## PART 2:
* Train classifier(s) (which classifier is best and why)
* Explain potential overfitting
* Explain potential bias
* How do we deal with bias and overfitting?
* Cross-validation why (not)?
## PART 3:
* Evaluate results
* Which metrics are (not) useful?
* Which conclusions can we (not) make?
* Maybe give some further recommendations on how to improve? (Like more data, different ML techniques etc.)
