import json
import random
import contractions
import re
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load JSON data
file_path = './data/pet_supplies.json'  # Update the path as needed
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract a random subset of reviews (100-1000)
sample_size = random.randint(100, 1000)
sample_data = random.sample(data, k=sample_size)

# Extract reviews and ratings
reviews = [item['review'] for item in sample_data]
ratings = [item['rating'] for item in sample_data]

# Save the extracted subset to a new JSON file
output_file_path = './data/selected_reviews.json'  # Update the path as needed
with open(output_file_path, 'w') as file:
    json.dump(sample_data, file, indent=4)

# Clean the text
def clean_text(text):
    # Fix contractions
    text = contractions.fix(text)
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f"[{re.escape(punctuation)}]", '', text)
    # Remove numbers
    text = re.sub(r'\w*\d\w*', '', text)
    # Remove stopwords
    stopwords_path = './Machinelearning/data/stopwords.txt'  # Update path
    with open(stopwords_path, 'r') as file:
        stopwords = {line.strip() for line in file}
    return ' '.join(word for word in text.split() if word not in stopwords)

# Apply cleaning to all reviews
cleaned_reviews = [clean_text(review) for review in reviews]

# Convert text to numerical representation
vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(cleaned_reviews)

# Split into training and testing sets (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, ratings, test_size=0.2, random_state=42)

# Train an SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Test the classifier
accuracy = clf.score(X_test, y_test)

# Output results
print(f"Model Accuracy: {accuracy * 100:.2f}%")
