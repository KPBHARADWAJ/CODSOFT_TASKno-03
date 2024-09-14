import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Sample dataset
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    'movie_id': [1, 2, 3, 1, 3, 2, 3, 4, 1, 4],
    'rating': [5, 4, 3, 4, 5, 2, 4, 5, 3, 4]
}

df = pd.DataFrame(data)

# Create a user-item matrix
user_item_matrix = df.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)

# Normalize the data
scaler = StandardScaler()
user_item_matrix_scaled = scaler.fit_transform(user_item_matrix)

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix_scaled)

# Convert similarity matrix to DataFrame
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def get_recommendations(user_id, num_recommendations=2):
    # Get the user's ratings
    user_ratings = user_item_matrix.loc[user_id]
    
    # Get the similarity scores for the user
    similarity_scores = user_similarity_df[user_id]
    
    # Compute the weighted sum of ratings for each movie
    weighted_sum = user_item_matrix.T.dot(similarity_scores)
    
    # Normalize by the sum of the similarity scores
    recommendation_scores = weighted_sum / similarity_scores.sum()
    
    # Exclude movies the user has already rated
    recommendations = recommendation_scores[user_ratings == 0]
    
    # Get the top N recommendations
    top_recommendations = recommendations.nlargest(num_recommendations)
    
    return top_recommendations.index.tolist()

user_id = 1
recommendations = get_recommendations(user_id)
print(f"Recommendations for user {user_id}: {recommendations}")
