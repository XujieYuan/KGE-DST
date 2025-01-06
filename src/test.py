from sentence_transformers import SentenceTransformer
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from llama_cpp import Llama
from neo4j import GraphDatabase
import numpy as np
import pickle
import re

# Initialize Neo4j
uri = "neo4j+s://???.databases.neo4j.io"
username = "neo4j"
password = "password"
driver = GraphDatabase.driver(uri, auth=(username, password))

# Load entity embeddings
with open('/data1/yuanxujie/KGE-DST/data/entity_embeddings.pkl', 'rb') as f:
    entity_embeddings = pickle.load(f)

entity_embeddings_emb = np.array(entity_embeddings["embeddings"])
entity_names = entity_embeddings["entities"]

# Initialize Sentence Transformer
model = SentenceTransformer('/data1/yuanxujie/pretrained_models/distiluse-base-multilingual-cased-v1')

# Initialize Chat Model
chat = ChatOpenAI(openai_api_key="EMPTY", openai_api_base="http://localhost:8000/v1", max_tokens=12800, temperature=0.0)

def chat_llm(prompt, n_gpu_layers=200, n_batch=512, n_ctx=25600, repeat_penalty=1.0, verbose=False):
    llm = Llama(
        model_path="/home/yuanxj/llama2/llama.cpp/models/7B/ggml-model-q4_0.bin",
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=n_ctx,
        chat_format="llama-2",
        repeat_penalty=repeat_penalty,
        verbose=verbose
    )
    completion = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant. Extract the main keyword from the following sentence. Output in 'KEYWORD:...' format"},
            {"role": "user", "content": prompt}
        ],
    )
    return completion["choices"][0]["message"]["content"]

# Helper Functions
def encode_text(text):
    embeddings = model.encode([text], convert_to_numpy=True)
    return embeddings[0]

def cosine_similarity_manual(x, y):
    dot_product = np.dot(x, y.T)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    sim = dot_product / (norm_x * norm_y)
    return sim

def get_entity_neighbors(entity_name):
    try:
        with driver.session() as session:
            query = """
                MATCH (e:Entity)-[r]->(n)
                WHERE e.name = $entity_name
                RETURN n.name AS neighbor, type(r) AS relationship_type
            """
            result = session.run(query, entity_name=entity_name)
            return [(record["neighbor"], record["relationship_type"]) for record in result]
    except Exception as e:
        print(f"Neo4j error: {e}")
        return []

def retrieve_context(question):
    input_embeddings = encode_text(question)
    cos_similarities = np.array([cosine_similarity_manual(input_embeddings, emb) for emb in entity_embeddings_emb])
    top_indices = np.argsort(cos_similarities)[::-1][:1]  # Top match

    context = []
    for idx in top_indices:
        entity = entity_names[idx]
        neighbors = get_entity_neighbors(entity)
        context.append((entity, neighbors))
    return context

def translate(input_text, source_lang, target_lang):
    # Step 1: Extract Keywords
    keywords = chat_llm(input_text)
    keyword = re.search(r"KEYWORD:(.+)", keywords).group(1).strip()

    # Step 2: Retrieve Knowledge Context
    context = retrieve_context(keyword)
    print(f'Context: {context}')

    # Step 3: Generate Translation
    if context:
        neighbors = context[0][1]
        # Select evidence dynamically based on target language
        evidence = next((neighbor for neighbor, relationship in neighbors if relationship.lower() == target_lang.lower()), "No relevant entity found.")
    else:
        evidence = "No relevant entity found."

    translation_prompt = [
        SystemMessage(
            content="You are a multilingual AI assistant. Translate the following sentence while considering the context provided."),
        AIMessage(content=f"Source: {input_text}\nEvidence: {evidence}\nTranslate from {source_lang} to {target_lang}."),
        HumanMessage(content="Output the final translation result as OUTPUT, as the following format: OUTPUT: translated text.")
    ]
    translation_response = chat(translation_prompt).content

    # Extract translation using regex
    match = re.search(r"OUTPUT:(.+)", translation_response)
    if match:
        formatted_translation = match.group(1).strip()
    else:
        formatted_translation = "Translation not found in expected format."

    print("Input Sentence:", input_text)
    print("Extracted Keywords:", keyword)
    print("Evidence Entity:", evidence)
    print("Translation:", formatted_translation)

if __name__ == "__main__":
    input_text = "格林-巴利综合征是一种急性自身免疫性周围神经系统疾病"
    source_lang = "Chinese"
    target_lang = "French"
    translate(input_text, source_lang, target_lang)
