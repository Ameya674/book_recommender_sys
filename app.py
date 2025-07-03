# importing necessary dependencies
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma  # ✅ Correct import
import gradio as gr


# loading the env variables
# load_dotenv()

# load the dataset
books = pd.read_csv("books-with-emotions.csv")

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(books["thumbnail"].isna(), "cover-not-found.jpeg", books["large_thumbnail"])

# initialize or load the vector database
VECTOR_DB_DIR = "vector_store"

if os.path.exists(VECTOR_DB_DIR):
    # load existing database
    vector_database = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=OpenAIEmbeddings())
else:
    # data preprocessing
    raw_documents = TextLoader("descriptions.txt").load()  # load the descriptions
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0) 
    documents = text_splitter.split_documents(raw_documents)  # split individual documents
    vector_database = Chroma.from_documents(documents, OpenAIEmbeddings(), persist_directory=VECTOR_DB_DIR)  # create a vector database using OpenAIEmbeddings
    vector_database.persist()


def retreive_semantic_recommendations(
    query: str, category: str=None, 
    tone: str=None, 
    initial_top_k: int=50, 
    final_top_k: int=16) -> pd.DataFrame:

    recommendations = vector_database.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recommendations]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    [int(rec.page_content.strip('"').split()[0]) for rec in recommendations]

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
    query: str,
    category: str,
    tone: str
):
    recommendations = retreive_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]


with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Please tell us what you want to read...")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category: ", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select a tone: ", value="All")
        submit = gr.Button("Find me a book!")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended Books", columns=4, rows=2)  # centered layout

    submit.click(fn=recommend_books,
                 inputs=[user_query, category_dropdown, tone_dropdown], outputs=output)


if __name__ == "__main__":
    dashboard.launch()
