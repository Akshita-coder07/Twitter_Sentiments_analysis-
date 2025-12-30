import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from ntscraper import Nitter

# Download stopwords and initialize stemmer once
@st.cache_resource
def load_nltk_resources():
    nltk.download('stopwords')
    return stopwords.words('english'), PorterStemmer()

# Load model and vectorizer once
@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Define sentiment prediction function (with stemming for consistency)
def predict_sentiment(text, model, vectorizer, stop_words, stemmer):
    # Preprocess text (match notebook)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    text = ' '.join(text)
    text = [text]
    text = vectorizer.transform(text)
    
    # Predict sentiment
    sentiment = model.predict(text)
    return "Negative" if sentiment == 0 else "Positive"

# Initialize Nitter scraper
@st.cache_resource
def initialize_scraper():
    return Nitter(log_level=1)

# Function to create a colored card
def create_card(tweet_text, sentiment):
    color = "lightgreen" if sentiment == "Positive" else "lightcoral"
    card_html = f"""
    <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0; border: 1px solid #ddd;">
        <h5>{sentiment} Sentiment</h5>
        <p>{tweet_text}</p>
    </div>
    """
    return card_html

# Main app logic
def main():
    st.set_page_config(page_title="Twitter Sentiment Analyzer", page_icon="🐦")
    st.title("🐦 Twitter Sentiment Analysis")
    st.markdown("Analyze sentiment from text or fetch tweets!")

    # Load resources
    stop_words, stemmer = load_nltk_resources()
    model, vectorizer = load_model_and_vectorizer()
    scraper = initialize_scraper()

    # User input
    option = st.selectbox("Choose an option", ["Input text", "Get tweets from user"])
    
    if option == "Input text":
        text_input = st.text_area("Enter text to analyze sentiment")
        if st.button("Analyze"):
            if text_input.strip():
                with st.spinner("Analyzing..."):
                    sentiment = predict_sentiment(text_input, model, vectorizer, stop_words, stemmer)
                st.success(f"Sentiment: {sentiment}")
            else:
                st.warning("Please enter text.")

    elif option == "Get tweets from user":
        username = st.text_input("Enter Twitter username (without @)")
        if st.button("Fetch Tweets"):
            if username.strip():
                with st.spinner("Fetching tweets..."):
                    try:
                        tweets_data = scraper.get_tweets(username, mode='user', number=5)
                        if 'tweets' in tweets_data and tweets_data['tweets']:
                            for tweet in tweets_data['tweets']:
                                tweet_text = tweet.get('text', '')
                                if tweet_text:
                                    sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words, stemmer)
                                    card_html = create_card(tweet_text, sentiment)
                                    st.markdown(card_html, unsafe_allow_html=True)
                        else:
                            st.error("No tweets found. Check username or try again.")
                    except Exception as e:
                        st.error(f"Error fetching tweets: {str(e)}")
            else:
                st.warning("Please enter a username.")

if __name__ == "__main__":
    main()