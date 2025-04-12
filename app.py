import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import HTML

# Cache data loading
@st.cache_data
def load_data():
    filtered_df = pd.read_excel("filtered_df.xlsx")
    Books_df = pd.read_excel("Books_df.xlsx")
    return filtered_df, Books_df

# Cache matrix computation
@st.cache_resource
def get_user_item_matrices(filtered_df):
    user_item_matrix = filtered_df.pivot_table(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)
    item_sim_matrix = pd.DataFrame(
        cosine_similarity(user_item_matrix.T),
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )
    return user_item_matrix, item_sim_matrix

# Recommendation functions
def recommend_for_user(user_id, n=5):
    if user_id not in user_item_matrix.index:
        return []
    user_ratings = user_item_matrix.loc[user_id]
    rated_books = user_ratings[user_ratings > 0].index.tolist()
    scores = {}

    for book in rated_books:
        similar_books = item_sim_matrix[book].drop(labels=rated_books, errors='ignore')
        for similar_book, sim in similar_books.items():
            if similar_book not in scores:
                scores[similar_book] = 0
            scores[similar_book] += sim

    sorted_books = sorted(scores, key=lambda x: (
        scores[x],
        filtered_df[filtered_df['ISBN'] == x]['Book-Rating'].mean()
    ), reverse=True)
    return sorted_books[:n]

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

def hybrid_recommend(user_id=None, book_title=None, n=5):
    if user_id and user_id in user_item_matrix.index:
        isbns = recommend_for_user(user_id, n)
        heading = f"\U0001F4DA Top {n} Recommendations for User ID {user_id}"
    elif book_title:
        isbns = recommend_for_book(book_title, n)
        heading = f"\U0001F4DA Top {n} Books Similar to '{book_title}'"
    else:
        avg_ratings = filtered_df.groupby('ISBN')['Book-Rating'].mean()
        count_ratings = filtered_df['ISBN'].value_counts()
        top_isbns = avg_ratings[count_ratings >= 20].sort_values(ascending=False).head(n).index.tolist()
        isbns = top_isbns
        heading = "\U0001F4DA Top Rated Books (Fallback)"

    if not isbns:
        st.warning("\u26A0\ufe0f No recommendations found.")
        return

    st.markdown(f"## {heading}")
    for isbn in isbns:
        book = Books_df[Books_df['ISBN'] == isbn]
        if book.empty:
            continue
        book = book.iloc[0]
        avg_rating = filtered_df[filtered_df['ISBN'] == isbn]['Book-Rating'].mean()
        st.image(book['Image-URL-M'], width=100)
        st.markdown(f"**{book['Book-Title']}**  ")
        st.markdown(f"Author: {book['Book-Author']}  ")
        st.markdown(f"Average Rating: {avg_rating:.2f}")
        st.markdown("---")

# Load data
filtered_df, Books_df = load_data()
user_item_matrix, item_sim_matrix = get_user_item_matrices(filtered_df)

# Streamlit UI
st.title("\U0001F4D6 Hybrid Book Recommendation System")

choice = st.radio("🔍 Recommend by:", ["User ID", "Book Title"])

if choice == "User ID":
    user_input = st.number_input("Enter User ID", min_value=0, step=1)
    if st.button("Get Recommendations"):
        hybrid_recommend(user_id=user_input)

elif choice == "Book Title":
    title_input = st.text_input("Enter Book Title")
    if st.button("Get Recommendations"):
        hybrid_recommend(book_title=title_input)

st.caption("Deployable version without Surprise library | Cached for fast performance")
