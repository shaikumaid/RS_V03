import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    filtered_df = pd.read_excel("filtered_df.xlsx")
    books_df = pd.read_excel("Books_df.xlsx")
    return filtered_df, books_df

filtered_df, Books_df = load_data()

# -----------------------------
# Preprocessing
# -----------------------------
user_item_matrix = filtered_df.pivot_table(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)

item_sim_matrix = pd.DataFrame(
    cosine_similarity(user_item_matrix.T),
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

Books_df['cleaned_title'] = Books_df['Book-Title'].str.strip().str.lower()

# -----------------------------
# Recommendation Functions
# -----------------------------
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
    best_match = process.extractOne(title.lower(), Books_df['cleaned_title'].tolist())

    if not best_match or best_match[1] < 70:
        return None

    matched_title = best_match[0]
    matched = Books_df[Books_df['cleaned_title'] == matched_title]
    if matched.empty:
        return None

    isbn = matched.iloc[0]['ISBN']
    if isbn not in item_sim_matrix:
        return None

    similar_scores = item_sim_matrix[isbn].drop(labels=[isbn])
    sorted_books = sorted(similar_scores.items(), key=lambda x: (
        x[1],
        filtered_df[filtered_df['ISBN'] == x[0]]['Book-Rating'].mean()
    ), reverse=True)

    return [isbn for isbn, _ in sorted_books[:n]]

def hybrid_recommend(user_id=None, book_title=None, n=5):
    isbns = []
    show_fallback = False
    fallback_message = None

    if user_id and user_id in user_item_matrix.index:
        isbns = recommend_for_user(user_id, n)
        heading = f"📚 Top {n} Recommendations for User ID {user_id}"

    elif user_id and user_id not in user_item_matrix.index:
        show_fallback = True
        fallback_message = (
            f"❗User ID '**{user_id}**' not found in the system. "
            f"Showing top-rated books instead."
        )
        heading = "📚 Top Rated Books"

    elif book_title:
        isbns = recommend_for_book(book_title, n)
        if isbns is None:
            show_fallback = True
            fallback_message = (
                f"❗Couldn’t find similar books for the title '**{book_title}**'. Please check and enter again "
                f"Showing top-rated books instead."
            )
            heading = "📚 Top Rated Books"
        else:
            heading = f"📚 Top {n} Books Similar to '{book_title}'"

    else:
        show_fallback = True
        heading = "📚 Top Rated Books (Fallback)"

    if show_fallback:
        avg_ratings = filtered_df.groupby('ISBN')['Book-Rating'].mean()
        count_ratings = filtered_df['ISBN'].value_counts()
        top_isbns = avg_ratings[count_ratings >= 20].sort_values(ascending=False).head(n).index.tolist()
        isbns = top_isbns

    if not isbns:
        st.warning("⚠️ No recommendations found.")
        return

    if fallback_message:
        st.warning(fallback_message)

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
        st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📚 Book Recommendation System")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.subheader("🔍 Recommend based on:")
    option = st.radio("", ["User ID", "Book Title"], horizontal=True)

    if option == "User ID":
        user_id = st.number_input("Enter User ID:", min_value=1, step=1)
        if st.button("Get Recommendations"):
            hybrid_recommend(user_id=int(user_id))

    else:
        st.markdown("You can either **type** a book title or **select** one from the dropdown.")
        typed_title = st.text_input("Type a book title:")

        top_isbns = filtered_df['ISBN'].value_counts().head(100).index.tolist()
        top_titles = Books_df[Books_df['ISBN'].isin(top_isbns)][['Book-Title']].dropna()
        book_options = top_titles['Book-Title'].drop_duplicates().sort_values().tolist()
        dropdown_title = st.selectbox("Or choose from dropdown:", [""] + book_options)

        if st.button("Get Recommendations"):
            title_to_use = typed_title if typed_title.strip() else dropdown_title
            if not title_to_use:
                st.warning("Please type or select a book title.")
            else:
                hybrid_recommend(book_title=title_to_use)
