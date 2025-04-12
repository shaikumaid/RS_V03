import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load data from Excel
@st.cache_data
def load_data():
    filtered_df = pd.read_excel("filtered_df.xlsx")
    books_df = pd.read_excel("Books_df.xlsx")
    return filtered_df, books_df

filtered_df, Books_df = load_data()

# Pivot to user-item matrix
user_item_matrix = filtered_df.pivot_table(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)

# Compute item similarity
item_sim_matrix = pd.DataFrame(
    cosine_similarity(user_item_matrix.T),
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

def recommend_for_user(user_id, n=5):
    if user_id not in user_item_matrix.index:
        return []
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
        heading = f"📚 Top {n} Recommendations for User ID {user_id}"
    elif book_title:
        isbns = recommend_for_book(book_title, n)
        heading = f"📚 Top {n} Books Similar to '{book_title}'"
    else:
        avg_ratings = filtered_df.groupby('ISBN')['Book-Rating'].mean()
        count_ratings = filtered_df['ISBN'].value_counts()
        top_isbns = avg_ratings[count_ratings >= 20].sort_values(ascending=False).head(n).index.tolist()
        isbns = top_isbns
        heading = "📚 Top Rated Books (Fallback)"

    if not isbns:
        st.warning("⚠️ No recommendations found.")
        return

    # Display Recommendations
    st.markdown(f"### {heading}")
    with st.spinner('Loading recommendations...'):
        cols = st.columns(2)  # Create two columns for a grid layout
        for i, isbn in enumerate(isbns):
            book = Books_df[Books_df['ISBN'] == isbn]
            if book.empty:
                continue
            book = book.iloc[0]
            avg_rating = filtered_df[filtered_df['ISBN'] == isbn]['Book-Rating'].mean()

            # Place each book in a different column for grid layout
            col = cols[i % 2]
            with col:
                # Display Book Title and Author in main view
                st.image(book['Image-URL-M'], width=120, use_container_width=True)  # Smaller image size
                st.markdown(f"**{book['Book-Title']}**")
                st.markdown(f"Author: {book['Book-Author']}")
                
                # Move the Average Rating to the "More Info" section
                with st.expander("More Info"):
                    st.write(f"Average Rating: {avg_rating:.2f}")
                    st.write(f"ISBN: {isbn}")

# Streamlit UI
st.title("📚 Book Recommendation System")

# Radio Button for choosing recommendation method
option = st.radio("🔍 Recommend based on:", ["User ID", "Book Title"])

if option == "User ID":
    user_id = st.number_input("Enter User ID:", min_value=1, step=1)
    if st.button("Get Recommendations"):
        hybrid_recommend(user_id=int(user_id))
else:
    title = st.text_input("Enter Book Title:")
    if st.button("Get Recommendations"):
        hybrid_recommend(book_title=title)
