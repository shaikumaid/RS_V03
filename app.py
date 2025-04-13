import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process  # Fuzzy matching

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
    # Clean and normalize book titles for comparison
    Books_df['cleaned_title'] = Books_df['Book-Title'].str.strip().str.lower()

    # Perform fuzzy matching to find similar books
    matches = process.extract(title.lower(), Books_df['cleaned_title'].tolist(), limit=5)

    if not matches:
        st.warning(f"No close match found for '{title}'. Showing top-rated fallback books.")
        return None  # Fallback if no match is found

    best_match = matches[0]  # Get the best match (most similar)
    matched_title = best_match[0]

    # Find all books that match this best fuzzy match
    matched_books = Books_df[Books_df['cleaned_title'] == matched_title]

    if matched_books.empty:
        st.warning(f"Could not find any matches for '{title}'. Showing top-rated fallback books.")
        return None  # If no match, show fallback books
    
    # Get the ISBN of the matched book
    isbn = matched_books.iloc[0]['ISBN']

    # Check if the book is part of the trained model (item similarity matrix)
    if isbn not in item_sim_matrix:
        st.warning(f"'{title}' was found but not in the trained model. Showing similar books based on title.")
        
        # Show books with similar titles from the dataset (fuzzy match results)
        similar_books = Books_df[Books_df['cleaned_title'].str.contains(matched_title, na=False)].head(n)
        return similar_books['ISBN'].tolist()  # Return ISBNs of similar books

    # If the book is in the trained model, show item-based similarity results
    similar_scores = item_sim_matrix[isbn].drop(labels=[isbn])  # Drop the original book from its similarity
    sorted_books = sorted(similar_scores.items(), key=lambda x: (
        x[1],
        filtered_df[filtered_df['ISBN'] == x[0]]['Book-Rating'].mean()
    ), reverse=True)

    return [isbn for isbn, _ in sorted_books[:n]]  # Return the top `n` books

def hybrid_recommend(user_id=None, book_title=None, n=5):
    isbns = []
    show_fallback = False

    if user_id and user_id in user_item_matrix.index:
        isbns = recommend_for_user(user_id, n)
        heading = f"📚 Top {n} Recommendations for User ID {user_id}"
    elif user_id and user_id not in user_item_matrix.index:
        st.error("❌ Invalid User ID. Please check and try again.")
        show_fallback = True
        heading = "📚 Top Rated Books (Fallback)"
    elif book_title:
        isbns = recommend_for_book(book_title, n)
        if isbns is None:
            show_fallback = True
            heading = "📚 Top Rated Books (Fallback)"
        else:
            heading = f"📚 Top {n} Books Similar to '{book_title}'"
    else:
        show_fallback = True
        heading = "📚 Top Rated Books (Fallback)"

    if show_fallback:
        # Fallback to top-rated books based on average ratings and a minimum of 20 ratings
        avg_ratings = filtered_df.groupby('ISBN')['Book-Rating'].mean()
        count_ratings = filtered_df['ISBN'].value_counts()
        top_isbns = avg_ratings[count_ratings >= 20].sort_values(ascending=False).head(n).index.tolist()
        isbns = top_isbns

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

        with st.container():
            cols = st.columns([1, 3])
            with cols[0]:
                st.image(book['Image-URL-M'], use_container_width=True)
            with cols[1]:
                st.markdown(f"**{book['Book-Title']}**")
                st.markdown(f"*Author: {book['Book-Author']}*")
                with st.expander("More Info", expanded=False):
                    st.markdown(f"Average Rating: **{avg_rating:.2f}**")
                    st.markdown(f"**ISBN:** {isbn}")

        # Adding space between books
        st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------
# UI Layout
# -----------------------------

st.title("📚 Book Recommendation System")

# Center layout for inputs
col1, col2, col3 = st.columns([1, 2, 1])  # Makes col2 centered

with col2:
    st.subheader("🔍 Recommend based on:")
    option = st.radio("", ["User ID", "Book Title"], horizontal=True)

    if option == "User ID":
        user_id = st.number_input("Enter User ID:", min_value=1, step=1)
        if st.button("Get Recommendations"):
            hybrid_recommend(user_id=int(user_id))

    else:
        # Top 100 most popular books for dropdown
        top_isbns = filtered_df['ISBN'].value_counts().head(100).index.tolist()
        top_titles = Books_df[Books_df['ISBN'].isin(top_isbns)][['Book-Title']].dropna()
        book_options = top_titles['Book-Title'].drop_duplicates().sort_values().tolist()

        title = st.selectbox("Choose or type a book title:", book_options)
        if st.button("Get Recommendations"):
            hybrid_recommend(book_title=title)
