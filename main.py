import streamlit as st
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from pymongo import MongoClient
import os
import config


# https://discuss.streamlit.io/t/message-error-about-torch/90886/3
#import torch
#torch.classes.__path__ = [] # add this line to manually set it to empty. 


uri = os.getenv("MONGO_CONNECTION")
client = MongoClient(uri)

collection = client[config.db][config.collection]

# Initialize text embedding model (encoder)
embeddings = HuggingFaceEmbeddings(model_name=config.hf_model)


index_name = config.index_name
vector_field_name = config.vector_field_name
text_field_name = config.text_field_name

# https://www.mongodb.com/developer/products/atlas/leveraging-mongodb-atlas-vector-search-langchain/
vectorStore = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name=index_name,
    embedding_key=vector_field_name,
    text_key=text_field_name,
)

class PromptLoggingCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        for i, prompt in enumerate(prompts):
            print(f"[Prompt Sent to LLM #{i}]:\n{prompt}\n")

callback_manager = CallbackManager([
    StreamingStdOutCallbackHandler(),
    PromptLoggingCallbackHandler()
])

llm = OllamaLLM(model=config.ollama_llm, callback_manager=callback_manager)

def main():
    st.title("Mongo RAG Document Chat")

    query = st.text_input("Enter your query:")

    retriever = vectorStore.as_retriever()

    # Query LLM with user input and context data
    if st.button("Query LLM"):
        with st.spinner("Querying LLM..."):

            # Retrieve the top relevant documents manually
            docs = retriever.get_relevant_documents(query)
        
            st.text("Retrieved Document IDs:")
            for doc in docs:
                st.text(f"- {doc.metadata.get("_id", "N/A")}")

            qa = RetrievalQA.from_chain_type(
                llm, chain_type="stuff", retriever=retriever
            )
            response = qa({"query": query})

            print(f"LLM Response: {response['result']}")
            st.text("Llama2 Response:")
            st.text(response["result"])



if __name__ == "__main__":
    main()
