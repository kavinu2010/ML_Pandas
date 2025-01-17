import json
import random
import contractions
import re
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load and sample data
with open('./data/pet_supplies.json', 'r') as f:
    data = json.load(f)

# Randomly select reviews between 100-1000
selected_reviews = random.sample(data, k=random.randint(100, 1000))

# Extract reviews and ratings
reviews = [item['review'] for item in selected_reviews]
ratings = [item['rating'] for item in selected_reviews]

# Save to a new file
subset_data = [{'review': review, 'rating': rating} for review, rating in zip(reviews, ratings)]
with open('./data/selected_reviews.json', 'w') as f:
    json.dump(subset_data, f)

# Clean text
def clean_text(text):
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(f'[{re.escape(punctuation)}]', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    stopwords = [line.strip() for line in open('./Machinelearning/data/stopwords.txt', 'r')]
    return ' '.join(word for word in text.split() if word not in stopwords)

cleaned_reviews = [clean_text(review) for review in reviews]

# Vectorize the text
vectorizer = CountVectorizer(ngram_range=(1, 2))
vectors = vectorizer.fit_transform(cleaned_reviews)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(vectors, ratings, test_size=0.2, random_state=42)

# Train an SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Test the classifier
predictions = clf.predict(X_test)
accuracy = clf.score(X_test, y_test)

print("Accuracy:", accuracy)
