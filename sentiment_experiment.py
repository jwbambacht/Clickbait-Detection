from dataset import Dataset

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

small_dataset = Dataset("datasets/small_training")


def get_sentiment(sentiment):
    compound = sentiment["compound"]
    if compound >= 0.5:
        return "positive"
    if compound <= -0.5:
        return "negative"
    return "neutral"

post_text = []
clickbait = []
for element in small_dataset.get_elements():
    post_text.append(" ".join(element.target_paragraphs))
    clickbait.append(element.get_truth().is_clickbait())


analyzer = SentimentIntensityAnalyzer()

clickbait_pos = 0
clickbait_neg = 0
clickbait_neu = 0

no_clickbait_pos = 0
no_clickbait_neg = 0
no_clickbait_neu = 0
for sentence, click in zip(post_text, clickbait):
    vs = analyzer.polarity_scores(sentence)
    sentiment = get_sentiment(vs)

    if click:
        if sentiment is "positive":
            clickbait_pos += 1
        if sentiment is "negative":
            clickbait_neg += 1
        if sentiment is "neutral":
            clickbait_neu += 1
    else:
        if sentiment is "positive":
            no_clickbait_pos += 1
        if sentiment is "negative":
            no_clickbait_neg += 1
        if sentiment is "neutral":
            no_clickbait_neu += 1

sum_click = clickbait_pos + clickbait_neg + clickbait_neu
print(f"Clickbait pos: {clickbait_pos /sum_click}")
print(f"Clickbait neg: {clickbait_neg /sum_click}")
print(f"Clickbait neu: {clickbait_neu /sum_click}")
print(f"---")
sum_no_click = no_clickbait_pos + no_clickbait_neg + no_clickbait_neu
print(f"Clickbait pos: {no_clickbait_pos / sum_no_click}")
print(f"Clickbait neg: {no_clickbait_neg /sum_no_click}")
print(f"Clickbait neu: {no_clickbait_neu /sum_no_click}")

import plotly.graph_objects as go
sentiments=['positive', 'neutral', 'negative']

fig = go.Figure(data=[
    go.Bar(name='Clickbait', x=sentiments, y=[(clickbait_pos /sum_click) * 100, (clickbait_neu /sum_click) * 100, (clickbait_neg /sum_click) * 100]),
    go.Bar(name='No-clickbait', x=sentiments, y=[(no_clickbait_pos / sum_no_click) * 100, (no_clickbait_neu / sum_no_click) * 100, (no_clickbait_neg / sum_no_click) * 100])
])
fig.update_yaxes(range=[0, 100])

# Change the bar mode
fig.update_layout(barmode='group')
fig.show()