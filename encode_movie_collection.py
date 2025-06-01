from sentence_transformers import SentenceTransformer
import pymongo
import os
from tqdm import tqdm
import config

mongo_uri = os.getenv("MONGO_CONNECTION")
db = config.db
collection = config.collection

connection = pymongo.MongoClient(mongo_uri)
collection = connection[db][collection]

# transofrmer model 
model = SentenceTransformer(config.hf_model)

# fields to include in embedding
fields_to_include = [
    "title", 
    "fullplot", 
    "plot", 
    "genres", 
    "directors", 
    "cast", 
    "writers", 
    "year", 
    "languages", 
    "rated", 
    "runtime", 
    "countries"
    ]

total = collection.count_documents({config.vector_field_name: {"$exists": False}})

for doc in tqdm(collection.find({config.vector_field_name: {"$exists": False}}), total=total, desc="Encoding"):
    movieid = doc["_id"]

    metadata_lines = []
    for field in fields_to_include:
        value = doc.get(field)
        if value:
            if isinstance(value, list):
                value_str = ", ".join(str(v) for v in value)
            else:
                value_str = str(value)
            metadata_lines.append(f"{field}: {value_str}")

    if not metadata_lines:
        continue

    full_text = "\n".join(metadata_lines)

    
    # compute embedding
    vector = model.encode(full_text).tolist()

    # update document with vector/embedding and the fulltext 
    collection.update_one(
        {"_id": movieid},
        {
            "$set": {
                config.vector_field_name: vector,
                config.text_field_name: full_text
            }
        },
        upsert=True
    )
    # print(f"Vector computed and stored for: {doc.get('title', movieid)}")

