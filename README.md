
# 🎬 MongoDB RAG Movie Chat with Streamlit, HuggingFace, and Ollama

A full-stack Retrieval-Augmented Generation (RAG) system built on top of MongoDB Atlas and LangChain. This app enables natural language queries over movie data using semantic search and a local LLM (Llama3 via Ollama).


## 📌 Project Overview

This project encodes the movies collection from the [sample-mflix](https://www.mongodb.com/docs/atlas/sample-data/sample-mflix/) dataset (title, plot, genre, cast, etc.) into vector embeddings using a HuggingFace Transformer Model , stores them in MongoDB Atlas with vector search enabled, and enables semantic search using a local LLM (run locally via [ollama](https://ollama.com/)) and [Streamlit] app as a frontend (https://streamlit.io/).

🔍 How It Works:

1. Data Ingestion: Pulls sample movie data from MongoDB Atlas.

2. Embedding Generation: Converts movie metadata into dense vectors using Sentence Transformers.

3. Vector Search: Stores these embeddings in a MongoDB Atlas collection with a vector index.

4. RAG Interface: A Streamlit frontend lets users ask questions in natural language. LangChain retrieves relevant movie documents using vector search, then passes them to the LLM for answer generation.


## 🛠️ Setup Instructions


## Configuration

The [config.py](config.py) file holds configurations for the used ollama llm, the hugginface model for embedding as well as the index names and db details 

```
db = "sample_mflix"
collection = "movies"

# index name created in mongo atlas 
index_name = "movie_embedding_index"
# field name in mongo docs that contain embedding vector 
vector_field_name = "movie_embedding_hf"
# field that contains text wich was embedded 
text_field_name = "embedding_fulltext"

ollama_llm = "llama3.2"

# https://huggingface.co/sentence-transformers/all-mpnet-base-v2
hf_model = "sentence-transformers/all-mpnet-base-v2"
vector_size = 768

# Small fast embedding model
# hf_model = "sentence-transformers/all-MiniLM-L6-v2"
# vector_size = 384
```


### 1. Create and Activate Virtual Environment

```
uv venv
uv pip install -r requirements.txt
source .venv/bin/activate
```
Or use ```python -m venv venv && source venv/bin/activate``` if you're not using uv.

### 2. Load Sample Data in MongoDB Atlas

Create a mongodb atlas account, log in and import the [sample-mflix](https://www.mongodb.com/docs/atlas/sample-data/sample-mflix/) dataset. The dataset already contains embedded_movies collection with embeddings for the plot field. This was embedded by 


### 3. Create Vector Search Index in MongoDB Atlas

Go to your MongoDB Atlas cluster and create a Vector Search Index named movie_embedding_index on the movie_embedding_hf field:

```
{
  "fields": [
    {
      "path": "movie_embedding_hf",
      "type": "vector",
      "numDimensions": 768,
      "similarity": "cosine"
    }
  ]
}
```

### 4. Configure Environment Variables

Export your MongoDB connection string:

```
export MONGO_CONNECTION="mongodb+srv://<username>:<password>@<cluster-url>"
```

### 🧠 Run Embedding Script
This script encodes metadata fields into vector embeddings and stores them in your MongoDB Atlas collection.

```
python encode_movie_collection.py
```

### ✅ Verifying Vectorized Documents

To check which documents have already been embedded, use this MongoDB query:

```
{ movie_embedding_hf: { $exists: true } }
```

### 🤖 Start the LLM Backend (Ollama)

Make sure Ollama is installed and run your local LLM (e.g., LLaMA 3):

```
ollama run llama3.2
```

### 🚀 Launch the Streamlit Application
```
streamlit run main.py
```

This opens a web app where you can enter natural language queries and see relevant movie data retrieved and answered by the LLM.


## 🧪 Example Queries

### Thematic or Genre-Based
- "Find action-packed movies involving espionage or spies."
- "What are some heartwarming family dramas?"
- "List sci-fi movies that explore artificial intelligence."
- "Give me psychological thrillers from the 2000s."
- "Find romantic comedies set in New York."

### Cast/Director-Based
- "Movies where Tom Hanks played a lead role."
- "Show films directed by Christopher Nolan that involve time travel."
- "What are some ensemble cast movies with Morgan Freeman?"

### Temporal Queries
- "What are some top-rated movies from the 90s?"
- "List recent horror films released after 2015."

### Awards/Critical Reception
- "Which movies won Oscars for Best Picture?"
- "Find critically acclaimed indie films."
- "What are some underrated movies with high ratings but low popularity?"

### Language or Country Specific
- "Foreign language thrillers from South Korea."
- "French romantic films from the 2010s."

### Plot/Content Based
- "Movies about survival in the wilderness."
- "Films that explore themes of identity and memory."
- "Movies based on true crime stories."
- "What are some movies where the protagonist is a robot or AI?"
- "Films where the ending is open to interpretation."


# Resources

* https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview/
