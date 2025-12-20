import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

@app.route('/')
def home():
    return render_template('index.html')

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the dataset
df = pd.read_excel('info_data_final.xlsx')
df = df.dropna(subset=['course'])

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r'\W+', ' ', str(text).lower())
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Preprocess necessary columns
df['publications_processed'] = df['pub'].apply(preprocess_text)
df['course_name_processed'] = df['name'].apply(preprocess_text)
df['course_processed'] = df['course'].apply(preprocess_text)
df['Keywords_processed'] = df['Keywords'].apply(preprocess_text)

# Include challenge level in combined features for similarity calculations
df['challenge_level_processed'] = df['Difficulty Category']

df['combined_features'] = df['course_name_processed'] + " " + df['publications_processed'] + " " + df['Keywords_processed'] + " " + df['challenge_level_processed']

# Recommendation function
def recommend_professors(text, course_code, challenge_level):
    # Filter for relevant professors based on course code
    filtered_professors = df[df['course'].str.contains(course_code, case=False)]
    filtered_professors = filtered_professors[filtered_professors['Difficulty Category'].str.contains(challenge_level, case=False)]
    print(filtered_professors)
    filtered_professors['combined_features'].fillna('', inplace=True)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_professors['combined_features'])

    user_input = preprocess_text(text + " " + challenge_level)
    print(user_input)
    user_vector = tfidf_vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)

    filtered_professors['Cosine Similarity'] = similarity_scores.flatten()

    features = filtered_professors[['Cosine Similarity', 'Course Difficulty Index','SFI','PEI']].fillna(0)
    features['Cosine Similarity'] *= 2

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    knn_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
    knn_model.fit(features_scaled)
    distances, indices = knn_model.kneighbors(features_scaled)

    recommended_professors = filtered_professors.iloc[indices[0]]

    recommended_professors = recommended_professors[recommended_professors['Cosine Similarity'] > 0]  # Filter non-zero similarity scores
    recommended_professors = recommended_professors.sort_values(by=['Cosine Similarity'], ascending=[False])    

    top_professors = []
    seen_professors = set()

    for _, row in recommended_professors.iterrows():
        if row['name'] not in seen_professors:
            seen_professors.add(row['name'])
            top_professors.append(row)
        if len(top_professors) > 4:
            break
    
    # Return as a DataFrame instead of dictionary
    return pd.DataFrame(top_professors)[['name', 'course', 'Cosine Similarity', 'SFI', 'PEI']]


# Flask route to handle recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    description = data.get('description', '')
    subject_code = data.get('subjectCode', '')
    challenge_level = data.get('challengeLevel', '')

    try:
        recommended_professors = recommend_professors(description, subject_code, challenge_level)
        
        recommendations_df = recommended_professors # Convert to DataFrame here
        
        # Check if Cosine Similarity values are preserved
        print("DataFrame with Recommendations:")
        print(recommendations_df[['name', 'course', 'Cosine Similarity']])

        # Ensure recommendations is a DataFrame
        if isinstance(recommendations_df, pd.DataFrame):
            relevant_recommendations = recommendations_df[recommendations_df['Cosine Similarity'] > -1]
            if relevant_recommendations.empty:
                return jsonify({"message": "No relevant recommendations available for the given description."}), 200

            # Convert DataFrame to dictionary for JSON response
            return jsonify({'recommendations': relevant_recommendations.to_dict(orient='records')})
        else:
            return jsonify({"error": "Unexpected data format in recommendations."}), 500

    except ValueError as e:
        if "empty vocabulary" in str(e):
            return jsonify({"error": "Invalid course ID or insufficient information. Please try a different input."}), 400
        else:
            return jsonify({"error": "An unexpected error occurred."}), 500

if __name__ == '__main__':
    app.run(debug=True)
