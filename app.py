import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load data
@st.cache_data
def load_data():
    filtered_df = pd.read_excel("filtered_df.xlsx")
    books_df = pd.read_excel("Books_df.xlsx")
    return filtered_df, books_df

filtered_df, Books_df = load_data()

# Create user-item matrix
user_item_matrix = filtered_df.pivot_table(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)

# Item similarity matrix
item_sim_matrix = pd.DataFrame(
    cosine_similarity(user_item_matrix.T),
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

# Recommend for user
def recommend_for_user(user_id, n=5):
    if user_id not in user_item_matrix.index:
        return None
    user_ratings = user_item_matrix.loc[user_id]
    rated_books = user_ratings[user_ratings > 0].index.tolist()
    scores = {}
    for book in rated_books:
        similar_books = item_sim_matrix[book].drop(labels=rated_books, errors='ignore')
        for similar_book, sim in similar_books.items():
            scores[similar_book] = scores.get(similar_book, 0) + sim
    sorted_books = sorted(scores, key=lambda x: (
        scores[x], 
        filtered_df[filtered_df['ISBN'] == x]['Book-Rating'].mean()
    ), reverse=True)
    return sorted_books[:n]

# Recommend for book
def recommend_for_book(title, n=5):
    matched = Books_df[Books_df['Book-Title'].str.lower() == title.lower()]
    if matched.empty:
        return []
    isbn = matched.iloc[0]['ISBN']
    if isbn not in item_sim_matrix:
        return []
    similar_scores = item_sim_matrix[isbn].drop(labels=[isbn])
    sorted_books = sorted(similar_scores.items(), key=lambda x: (
        x[1], 
        filtered_df[filtered_df['ISBN'] == x[0]]['Book-Rating'].mean()
    ), reverse=True)
    return [isbn for isbn, _ in sorted_books[:n]]

# Hybrid recommend
def hybrid_recommend(user_id=None, book_title=None, n=5):
    if user_id and user_id in user_item_matrix.index:
        isbns = recommend_for_user(user_id, n)
        heading = f"📚 Top {n} Recommendations for User ID {user_id}"
    elif book_title:
        isbns = recommend_for_book(book_title, n)
        heading = f"📚 Top {n} Books Similar to '{book_title}'"
    else:
        if user_id:
            st.warning("⚠️ Wrong ID entered. Showing Top Rated Books!")
        avg_ratings = filtered_df.groupby('ISBN')['Book-Rating'].mean()
        count_ratings = filtered_df['ISBN'].value_counts()
        top_isbns = avg_ratings[count_ratings >= 20].sort_values(ascending=False).head(n).index.tolist()
        isbns = top_isbns
        heading = "📚 Top Rated Books (Fallback)"

    if not isbns:
        st.warning("⚠️ No recommendations found.")
        return

    st.markdown(f"### {heading}")
    
    for isbn in isbns:
        book = Books_df[Books_df['ISBN'] == isbn]
        if book.empty:
            continue
        book = book.iloc[0]
        avg_rating = filtered_df[filtered_df['ISBN'] == isbn]['Book-Rating'].mean()

        # Display row layout
        cols = st.columns([1, 4])
        with cols[0]:
            st.markdown(f"""
                <div class="zoom-container">
                    <img src="{book['Image-URL-M']}" alt="Book Cover">
                </div>
            """, unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"""
                <div style="font-size: 16px; line-height: 1.4;">
                    <b>{book['Book-Title']}</b><br>
                    Author: {book['Book-Author']}<br>
                    <details style="font-size: 14px; margin-top: 6px;">
                        <summary style="cursor: pointer;">More Info</summary>
                        ISBN: {book['ISBN']}<br>
                        Average Rating: {avg_rating:.2f}
                    </details>
                </div>
            """, unsafe_allow_html=True)

# App UI
st.set_page_config(page_title="Book Recommender", layout="centered")
st.title("📚 Book Recommendation System")

# Custom CSS for zoom effect and centering inputs
st.markdown("""
    <style>
        .zoom-container {
            overflow: hidden;
            width: 100px;
            height: 150px;
        }
        .zoom-container img {
            width: 100%;
            height: 100%;
            transition: transform 0.3s ease;
        }
        .zoom-container img:hover {
            transform: scale(1.5);
        }
        .centered {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .stRadio > div {
            display: flex;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

# Centered Input Section
st.markdown('<div class="centered">', unsafe_allow_html=True)
option = st.radio("🔍 Recommend based on:", ["User ID", "Book Title"])
st.markdown('</div>', unsafe_allow_html=True)

# Input & Button
if option == "User ID":
    st.markdown('<div class="centered">', unsafe_allow_html=True)
    user_id = st.number_input("Enter User ID:", min_value=1, step=1)
    if st.button("Get Recommendations"):
        hybrid_recommend(user_id=int(user_id))
    st.markdown('</div>', unsafe_allow_html=True)
else:
    top_books = Books_df['Book-Title'].value_counts().head(100).index.tolist()
    title = st.selectbox("Select or enter Book Title:", options=top_books, index=0)
    if st.button("Get Recommendations"):
        hybrid_recommend(book_title=title)
