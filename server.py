from flask import Flask, request, jsonify
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import joblib

# Load MinMaxScaler model
scaler = joblib.load('scaler_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
tfidf_matrix = joblib.load('tfidf_matrix.pkl')

app = Flask(__name__)
df=pd.read_csv('Final_data.csv')





# Endpoint for scaling data
@app.route('/scale', methods=['POST'])
def scale_data():
    user_profile = request.get_json()
    top_n = 5
    user_profile_scaled = scaler.transform([[user_profile['followers'], user_profile['minEng'], user_profile['est_cost']]])
    min_followers_scaled, min_engagement_scaled, max_cost_scaled = user_profile_scaled[0]
    # print(user_profile_scaled) 
    similarities = {}
    for index, row in df.iterrows():
        influencer_vector = [row['Followers'], row['EngAvg'], row['Est_Pay']]
        user_profile_vector = [min_followers_scaled, min_engagement_scaled, max_cost_scaled]
        similarities[row['influencer']] = 1 - cosine(influencer_vector, user_profile_vector)
    # print(similarities)
    sorted_influencers = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    response = {}
    for i in sorted_influencers:
        ind=np.where(df['influencer']==i[0])
        # print(df.iloc[ind]['Followers'], df.iloc[ind]['EngAvg'], df.iloc[ind]['Est_Pay'])
        print(df.iloc[ind]['Followers'], df.iloc[ind]['EngAvg'], df.iloc[ind]['Est_Pay'])
        response[i[0]]={'followers':float(df.iloc[ind]['Followers']), 'avg_Engagement':float(df.iloc[ind]['EngAvg']), 'Estimated_pay':float(df.iloc[ind]['Est_Pay'])}
    return jsonify({'suggested_Influencers': [response]}), 200




#Endpoint for catagorical matching
@app.route('/reccat', methods=['POST'])
def Catagorical():
    data = request.get_json()
    user_profile_text = data['country'] + ' ' + data['category']
    user_profile_vector = tfidf_vectorizer.transform([user_profile_text])
    
    similarity_scores = cosine_similarity(user_profile_vector, tfidf_matrix)
    similarity_scores = similarity_scores.flatten()

    top_indices = similarity_scores.argsort()[::-1][:5]

    recommended_influencers = df.iloc[top_indices]
    recommended_influencers = recommended_influencers.to_dict(orient='records')

    return jsonify(recommended_influencers), 200





@app.route('/recommend', methods=['POST'])
def Recommend():
    user_profile = request.get_json()
    top_n = 30
    user_profile_scaled = scaler.transform([[user_profile['followers'], user_profile['minEng'], user_profile['est_cost']]])
    min_followers_scaled, min_engagement_scaled, max_cost_scaled = user_profile_scaled[0]
    # print(user_profile_scaled) 
    similarities = {}
    for index, row in df.iterrows():
        influencer_vector = [row['Followers'], row['EngAvg'], row['Est_Pay']]
        user_profile_vector = [min_followers_scaled, min_engagement_scaled, max_cost_scaled]
        similarities[row['influencer']] = 1 - cosine(influencer_vector, user_profile_vector)
    # print(similarities)
    sorted_influencers = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    #----------------------------------------------------------------------------------------Catagorical------------------------------------------------------------
    data = user_profile
    user_profile_text = data['country'] + ' ' + data['category']
    user_profile_vector = tfidf_vectorizer.transform([user_profile_text])
    
    similarity_scores = cosine_similarity(user_profile_vector, tfidf_matrix)
    similarity_scores = similarity_scores.flatten()

    top_indices = similarity_scores.argsort()[::-1][:top_n]

    recommended_influencers = df.iloc[top_indices]
    recommended_influencers = recommended_influencers.to_dict(orient='records')

    response = {}
    for i in sorted_influencers:
        ind=np.where(df['influencer']==i[0])
        # print(df.iloc[ind]['Followers'], df.iloc[ind]['EngAvg'], df.iloc[ind]['Est_Pay'])
        print(df.iloc[ind]['Followers'], df.iloc[ind]['EngAvg'], df.iloc[ind]['Est_Pay'])
        response[i[0]]={'followers':float(df.iloc[ind]['Followers']), 'avg_Engagement':float(df.iloc[ind]['EngAvg']), 'Estimated_pay':float(df.iloc[ind]['Est_Pay'])}

    return jsonify({'suggested_Influencers': [response]}), 200

if __name__ == '__main__':
    app.run(debug=True)