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