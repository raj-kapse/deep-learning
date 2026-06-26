from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

listt = [
    "I love machine learning",
    "I hate boring movies",
    "Machine learning is amazing",
    "Movies are so boring"
]

Vectorizer = TfidfVectorizer()

X = Vectorizer.fit_transform(listt)
y = [1,0,1,0]

model = LogisticRegression()
model.fit(X,y)

text = [
    "Learning is amazing"
]

X_test = Vectorizer.transform(text)

print(model.predict(X_test))


from transformers import pipeline

classifier = pipeline("sentiment-analysis")

results = classifier([
    "I love machine learning",
    "This movie was absolutely terrible",
    "The weather is okay I guess"

])

for result in results:
    print(result)