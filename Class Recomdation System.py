import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ClassRecommendationSystem:
    def __init__(self):
        self.data = None
        self.tfidf_matrix = None
        self.indices = None
    
    def load_data(self, filepath):
        self.data = pd.read_csv(filepath)
    
    def preprocess_data(self):
        # Preprocess your data if necessary
        # For example, you can clean the text, handle missing values, etc.
        # Ensure that the data has a column for 'description' and 'class' information
    
    def build_tfidf_matrix(self):
        tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.data['description'])
        self.indices = pd.Series(self.data.index, index=self.data['class']).drop_duplicates()
    
    def get_similar_classes(self, input_description, num_recommendations=5):
        # Transform input description into TF-IDF features
        input_tfidf = self.tfidf_matrix.transform([input_description])
        
        # Compute the cosine similarity between the input and all classes
        cosine_similarities = linear_kernel(input_tfidf, self.tfidf_matrix).flatten()
        
        # Get the indices of the classes with highest similarity scores
        class_indices = cosine_similarities.argsort()[:-num_recommendations-1:-1]
        
        # Map the class indices to their respective class names
        recommended_classes = self.data['class'].iloc[class_indices].tolist()
        
        return recommended_classes
Restapi
from flask import Flask, request, jsonify

app = Flask(__name__)
rec_system = ClassRecommendationSystem()

@app.route('/recommend', methods=['POST'])
def recommend_classes():
    data = request.get_json()
    input_description = data['description']
    num_recommendations = data.get('num_recommendations', 5)
    
    recommended_classes = rec_system.get_similar_classes(input_description, num_recommendations)
    
    response = {
        'recommended_classes': recommended_classes
    }
    return jsonify(response)

if __name__ == '__main__':
    # Load and preprocess the data
    rec_system.load_data('data.csv')
    rec_system.preprocess_data()

    # Build the TF-IDF matrix
    rec_system.build_tfidf_matrix()

    # Run the Flask application
    app.run()
# Create an instance of ClassRecommendationSystem
rec_system = ClassRecommendationSystem()

# Load and preprocess the data
rec_system.load_data('data.csv')
rec_system.preprocess_data()

# Build the TF-IDF matrix
rec_system.build_tfidf_matrix()

# Get recommendations for an input description
input_description = "This is a description of a product"
recommended_classes = rec_system.get_similar_classes(input_description)

print(recommended_classes)
{
  "description": "This is a description of a product",
  "num_recommendations": 3
}
{
  "recommended_classes": [
    "Class A",
    "Class B",
    "Class C"
  ]
}
