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