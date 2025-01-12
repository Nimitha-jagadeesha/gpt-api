import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI

# Set OpenAI API key
client = OpenAI(api_key="api-key")

# Example Dataset
data = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago'],
    'Job': ['Engineer', 'Artist', 'Doctor']
})

# User semantic input (example)
user_input = "Looking for a 30-year-old in Los Angeles"

# Step 1: Preprocess Tabular Data
# Define transformations for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age']),  # Scale numerical data
        ('cat', OneHotEncoder(), ['City', 'Job'])  # One-hot encode categorical data
    ]
)

# Fit and transform the tabular data
X = data[['Age', 'City', 'Job']]  # Exclude non-feature columns like 'Name'
vectorized_data = preprocessor.fit_transform(X)

# Step 2: Create Textual Representation for Rows
# Combine relevant features into descriptive text
text_data = data.apply(lambda row: f"{row['Name']} is a {row['Age']}-year-old {row['Job']} from {row['City']}.", axis=1)

# Step 3: Generate OpenAI Embeddings
def get_openai_embedding(text):
    try:
        response = client.embeddings.create(input=text,
        model="text-embedding-ada-002")
        return response.data[0].embedding
    except Exception as e:
        print(f"Error fetching embedding for text '{text}': {e}")
        return None


# Generate embeddings for each row and user input
row_embeddings = np.array([get_openai_embedding(text) for text in text_data])
user_embedding = get_openai_embedding(user_input)

# Step 4: Compute Similarities
similarities = cosine_similarity([user_embedding], row_embeddings)[0]

# Step 5: Retrieve the Most Similar Rows
top_match_indices = np.argsort(similarities)[::-1][:3]  # Get top 3 matches
top_matches = data.iloc[top_match_indices]
top_scores = similarities[top_match_indices]

# Display Results
print("Top Matches:")
for i, (index, score) in enumerate(zip(top_match_indices, top_scores)):
    print(f"Rank {i + 1}:")
    print(f"Row: {data.iloc[index].to_dict()}")
    print(f"Similarity Score: {score:.2f}")
    print()
