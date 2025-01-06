# # # Count entities and relations
# def count_entities_and_relations(file_path):
#     entities = set()
#     relations = set()
    
#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             for line_number, line in enumerate(file, 1):
#                 parts = line.strip().split('\t')
#                 if len(parts) != 3:
#                     print(f"Warning: Skipping invalid line {line_number}: {line.strip()}")
#                     continue
                
#                 entities.add(parts[0])
#                 entities.add(parts[2])
#                 relations.add(parts[1])
        
#         print(f"Total unique entities: {len(entities)}")
#         print(f"Total unique relations: {len(relations)}")
        
#         return len(entities), len(relations)
    
#     except FileNotFoundError:
#         print(f"Error: File not found: {file_path}")
#     except Exception as e:
#         print(f"Error: An unexpected error occurred: {str(e)}")

# if __name__ == "__main__":
#     # file_path = "./data/CWQ/knowledge.txt"
#     file_path = "/data1/yuanxujie/KGE-DST/data/knowledge_graph.txt" 
#     count_entities_and_relations(file_path)
#####################################################################
# # # Encode
import json
import pickle
from sentence_transformers import SentenceTransformer


def load_entities(file_path):
    entities = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    print(f"Warning: Skipping invalid line {line_number}: {line.strip()}")
                    continue
                entities.add(parts[0].strip())
                entities.add(parts[2].strip())
        return list(entities)
    except FileNotFoundError:
        print(f"Error: Entity file not found: {file_path}")
        return []


def encode_and_save(items, model, output_file, item_type):
    try:
        embeddings = model.encode(items, batch_size=1024, show_progress_bar=True, normalize_embeddings=True)
        data = {
            f"{item_type}": items,  # 保留原始关键词或实体（包括空格）
            "embeddings": embeddings,
        }
        with open(output_file, "wb") as f:
            pickle.dump(data, f)
        print(f"Encoded and saved {len(items)} {item_type} to {output_file}")
    except Exception as e:
        print(f"Error: Failed to encode and save {item_type}: {str(e)}")


def main():
    model = SentenceTransformer('/data1/yuanxujie/pretrained_models/distiluse-base-multilingual-cased-v1')

    # CWQ
    # 加载实体和关键词
    entities = load_entities("/data1/yuanxujie/KGE-DST/data/knowledge_graph.txt")
    # 编码并保存实体
    encode_and_save(entities, model, "/data1/yuanxujie/KGE-DST/data/entity_embeddings.pkl", "entities")

if __name__ == "__main__":
    main()
######################################################################
# # # # TEST
# import json
# import pickle
# import numpy as np
# import pandas as pd

# def load_pkl(file_path):
#     try:
#         with open(file_path, 'rb') as f:
#             data = pickle.load(f)
#         print(f"Keys in {file_path}: {list(data.keys())}")
#         return data
#     except FileNotFoundError:
#         print(f"Error: File not found: {file_path}")
#         return None
#     except Exception as e:
#         print(f"Error loading {file_path}: {str(e)}")
#         return None


# def adjust_keys(data, expected_key):
#     if expected_key not in data:
#         possible_keys = [key for key in data.keys() if key != 'embeddings']
#         if possible_keys:
#             print(f"Warning: '{expected_key}' not found. Using '{possible_keys[0]}' instead.")
#             data[expected_key] = data[possible_keys[0]]
#         else:
#             print(f"Error: Unable to find a suitable key for {expected_key}")
#             return None
#     return data


# def cosine_similarity_manual(A, b):
#     dot_product = np.dot(A, b)
#     norm_A = np.linalg.norm(A, axis=1)
#     norm_b = np.linalg.norm(b)
#     return dot_product / (norm_A * norm_b)


# def match_keywords_to_entities(entity_embeddings, keyword_embeddings, keywords, threshold=0.5):
#     entity_embeddings_df = pd.DataFrame(entity_embeddings["embeddings"])
#     match_kg = []
#     for keyword in keywords:
#         if keyword not in keyword_embeddings["keywords"]:
#             print(f"Warning: '{keyword}' not found in keyword embeddings")
#             continue
        
#         keyword_index = keyword_embeddings["keywords"].index(keyword)
#         keyword_emb = np.array(keyword_embeddings["embeddings"][keyword_index])
        
#         cos_similarities = cosine_similarity_manual(entity_embeddings_df, keyword_emb)
#         max_index = cos_similarities.argmax()
        
#         if cos_similarities[max_index] < threshold:
#             continue
        
#         match_entity = entity_embeddings["entities"][max_index]
#         while match_entity in match_kg:
#             cos_similarities[max_index] = 0
#             max_index = cos_similarities.argmax()
#             match_entity = entity_embeddings["entities"][max_index]
        
#         match_kg.append(match_entity)
    
#     return match_kg


# def custom_keyword_test(entity_embeddings, keyword_embeddings, custom_keywords):
#     print("\n--- Custom Keyword Test ---")
#     # print(f"Keywords in keyword embeddings: {keyword_embeddings['keywords']}")
#     match_kg = match_keywords_to_entities(entity_embeddings, keyword_embeddings, custom_keywords)
#     print(f"Custom Keywords: {custom_keywords}")
#     print(f"Matched Entities: {match_kg}")


# def json_sample_test(entity_embeddings, keyword_embeddings, json_file, num_samples=5):
#     print(f"\n--- JSON Sample Test (Testing {num_samples} samples) ---")
#     try:
#         with open(json_file, 'r', encoding='utf-8') as f:
#             data = [json.loads(line) for line in f.readlines()[:num_samples]]
#     except FileNotFoundError:
#         print(f"Error: JSON file not found: {json_file}")
#         return
#     except json.JSONDecodeError:
#         print(f"Error: Invalid JSON in file: {json_file}")
#         return

#     for sample in data:
#         question = sample.get('question', 'No question available')
#         question_kg = sample.get('question_kg', '').split(',')
#         question_kg = [kg.strip() for kg in question_kg]
        
#         print(f"\nQuestion: {question}")
#         print(f"Question KG: {question_kg}")
        
#         match_kg = match_keywords_to_entities(entity_embeddings, keyword_embeddings, question_kg)
#         print(f"Matched KG: {match_kg}")


# def main():
#     entity_embeddings = load_pkl("./data/CWQ/entity_embeddings.pkl")
#     keyword_embeddings = load_pkl("./data/CWQ/keyword_embeddings.pkl")
#     json_file = "./data/CWQ/cwq_with_keywords.json"

#     if entity_embeddings is None or keyword_embeddings is None:
#         print("Error: Failed to load embeddings. Exiting.")
#         return

#     entity_embeddings = adjust_keys(entity_embeddings, 'entities')
#     keyword_embeddings = adjust_keys(keyword_embeddings, 'keywords')

#     if entity_embeddings is None or keyword_embeddings is None:
#         print("Error: Failed to adjust keys. Exiting.")
#         return

#     print(f"Loaded {len(entity_embeddings['entities'])} entities and {len(keyword_embeddings['keywords'])} keywords.")

#     # Custom keyword test
#     custom_keywords = ["Super Bowl", "win", "Adolf Hitler"]
#     custom_keyword_test(entity_embeddings, keyword_embeddings, custom_keywords)

#     # JSON sample test
#     json_sample_test(entity_embeddings, keyword_embeddings, json_file, num_samples=2)

# if __name__ == "__main__":
#     main()