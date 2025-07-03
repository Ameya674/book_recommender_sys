# Semantic Book Recommender

An AI-powered book recommendation system using OpenAI embeddings, LangChain, and Gradio for an interactive user experience.

## Features

- Semantic search for personalized book recommendations  
- Interactive Gradio frontend  
- Efficient vector storage using ChromaDB  

## Setup Instructions

1. **Clone the repository**  
```bash
git clone https://github.com/YourGitHubUsername/BookRecommender.git
cd BookRecommender

```bash
python3 -m venv env
source env/bin/activate   
pip install -r requirements.txt
```

## Create a .env file in the project root with the following content:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## Run the app
```bash
python gradio-dashboard.py
```

