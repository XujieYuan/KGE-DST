import time
import heapq
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import LlamaCpp
from llama_cpp import Llama
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
import numpy as np
import re
import string
from neo4j import GraphDatabase, basic_auth
from neo4j.exceptions import ServiceUnavailable
import pandas as pd
from collections import deque
import itertools
from typing import Dict, List, Tuple
import pickle
import json
import csv
from time import sleep
import datetime
import os

from transformers import BertTokenizer, BertModel
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

def chat_llm(prompt, model_path, chat_format=None, stop=None, n_gpu_layers=200, n_batch=512, n_ctx=25600, repeat_penalty=1.0, verbose=False):
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=n_ctx,
        chat_format=chat_format,
        repeat_penalty=repeat_penalty,
        verbose=verbose
    )
    completion = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant. Please answer the following questions."},
            {"role": "user", "content": prompt}
        ],
        stop=stop  # 加入可选的 stop 参数
    )
    return completion["choices"][0]["message"]["content"]


# 英文 LLMs
def chat_llama2_7B(prompt):
    return chat_llm(prompt, "/home/yuanxj/llama2/llama.cpp/models/7B/ggml-model-q4_0.bin", chat_format="llama-2")


def encode_text(text):
    tokens = bert_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings


def extract_keywords_prompt(question: str, max_keywords: int = 1) -> str:
    return f"""Extract the most critical keywords from the given question. Your response must:
1. Start with "KEYWORDS:" followed by ONLY a comma-separated list of keywords
2. Include no more than {max_keywords} keywords
3. Include no explanation or other text

Example input: "What year did the San Francisco Giants last win the World Series?"
Example output: KEYWORDS: San Francisco Giants,World Series,win

Question: {question}"""


def extract_keywords(question: str, max_keywords: int = 7, max_retries: int = 5) -> str:
    prompt = extract_keywords_prompt(question, max_keywords)
    for attempt in range(max_retries):
        try:
            response = chat(prompt)
            print(f"LLM output:\n{response}")
            
            # Look for keywords after "KEYWORDS:" marker
            match = re.search(r'KEYWORDS:\s*([^"\n]+(?:\s*,\s*[^"\n]+)*)', response, re.IGNORECASE)
            if match:
                # Clean up the extracted keywords
                keywords = match.group(1).strip()
                # Remove any quotes and clean up spaces around commas
                keywords = re.sub(r'\s*,\s*', ',', keywords)
                keywords = keywords.replace('"', '').strip()
                print(f"Extracted keywords: {keywords}")
                return keywords
            else:
                print(f"Attempt {attempt + 1}: No valid keywords found, retrying...")
                time.sleep(1)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            time.sleep(1)

    print("Max retries reached. No valid keywords extracted.")
    return ""


def preprocess_output(output):
    # 删除多余的换行符
    output = re.sub(r"[\n\r]+", " ", output)
    
    # 将逗号替换为句号
    output = output.replace(",", ".")
    
    # 将冒号和句号替换为空格
    output = re.sub(r"[:.]", " ", output)
    
    # 将多个连续的空格替换为单个空格
    output = re.sub(r"\s+", " ", output)
    
    # 删除句首和句尾的空格
    output = output.strip()
    
    return output


def cosine_similarity_manual(x, y):
    dot_product = np.dot(x, y.T)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    sim = dot_product / (norm_x[:, np.newaxis] * norm_y)
    return sim


def calculate_score_noTopic(entity, input_question):
    """
    计算输入问题和实体之间的相似度得分。
    """
    input_question_embeddings = encode_text(input_question)
    entity_embeddings = encode_text(entity)
    similarity_score = torch.cosine_similarity(input_question_embeddings, entity_embeddings, dim=0).item()
    
    return similarity_score


def retrieve_top_triples(input_question, match_kg, top_n=3):
    """
    直接根据 match_kg 实体与问题的相似度，检索与问题最相关的 top_n 个三元组
    """
    top_triples = []
    
    # 遍历 match_kg 中的实体
    for entity in match_kg:
        # 从 Neo4j 数据库中检索该实体的所有邻居及其关系
        try:
            neighbors = get_entity_neighbors(entity)
        except ServiceUnavailable:
            print(f"Failed to retrieve neighbors for entity: {entity}")
            continue
        
        # 对每个邻居计算三元组的相似度
        for neighbor, relationship in neighbors:
            triple_text = f"{entity} {relationship} {neighbor}"
            score = calculate_score_noTopic(triple_text, input_question)
            top_triples.append((score, (entity, relationship, neighbor)))
    
    # 按得分从高到低排序并选择前 top_n 个三元组
    top_triples.sort(reverse=True, key=lambda x: x[0])
    
    # 返回最相关的 top_n 个三元组
    return [triple for _, triple in top_triples[:top_n]]


def get_entity_neighbors(entity_name: str) -> List[Tuple[str, str]]:
    global driver
    try:
        with driver.session() as session:
            query = """
                MATCH (e:Entity)-[r]->(n)
                WHERE e.name = $entity_name
                RETURN n.name AS neighbor, type(r) AS relationship_type
            """
            result = session.run(query, entity_name=entity_name)
            
            neighbor_list = []
            for record in result:
                relationship_type = record["relationship_type"]
                neighbor = record["neighbor"]
                neighbor_list.append((neighbor, relationship_type))
            
            return neighbor_list
    except ServiceUnavailable as e:
        print("Database connection lost. Attempting to reconnect...")
        try:
            driver.close()
            driver = GraphDatabase.driver(uri, auth=(username, password))
            return get_entity_neighbors(entity_name)
        except ServiceUnavailable:
            print("Failed to reconnect to the database.")
            raise


def final_output(str, evidence):
    messages = [
        SystemMessage(
            content="You are an excellent AI question-answering assistant that can provide Answer based on relevant knowledge of the Question."
        ),
        HumanMessage(content="Question: " + input_text),
        AIMessage(
            content="You have some relevant knowledge information in the following:\n\n### " + evidence 
        ),
        HumanMessage(
            content=
                "What is the answer of the question? "
                +"Think step by step.\n\n\n"
                +"The output should contain three parts: 'Output1, Output2, Output3'.\n\n"
                +"Output1: The answer of the input question.\n\n"
                +"Output2: Show the inference process as a string, extracting relevant knowledge from which Path-based Evidence or Neighbor-based Evidence, and infer the result. "
                +"Output3: Draw a decision tree. The entity or relation in single quotes in the inference process is added as a node with the source of evidence, "
                +"There is an example(do not copy the example):\n"
                +"""
INPUT: What's the currency used in the country where the Jamaican dollar as money is called?

OUTPUT: Jamaican dollar
                """
            )
    ]
    # Add a try/except block to handle the exception
    attempt_count = 0
    max_attempts = 3  # 设置一个最大尝试次数
    while attempt_count < max_attempts:
        try:
            # 尝试执行原本的功能
            result = chat(messages)
            output_all = result.content
            print("generate mindmap final output success!")
            return output_all  
        except Exception as e:
            # 如果遇到异常，等待一段时间后重试
            print(f"An error occurred in output_all: {e}. ")
            time.sleep(5)  # 等待5秒
            attempt_count += 1   
    return "Request failed after multiple attempts."  # 在连续失败后返回错误信息


def generate_output_all(input_text, evidence):
    try:
        output_all = final_output(input_text, evidence)
    except Exception as e:
        print(f"An error occurred: {e}. Failed to generate final answer")
        return None
    return output_all
    

if __name__ == "__main__":
    # 1. build neo4j knowledge graph datasets
    uri = "neo4j+s://6f2b7e4b.databases.neo4j.io"
    username = "neo4j"
    password = "g4ujIOCRsMIVCGOEhDXG0OiK04ZVdTobkfW1fz6yIdI"
    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session()

    # Initialize the BERT model and tokenizer for bi-encoder
    BERT_PATH = '/data1/yuanxujie/pretrained_models/bert-base-uncased'
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    bert_model = BertModel.from_pretrained(BERT_PATH)

    # # 2. OpenAI API based chat
    chat = ChatOpenAI(openai_api_key="EMPTY", openai_api_base="http://localhost:8000/v1", max_tokens=12800, temperature=0.0)


    with open('./data/CWQ/entity_embeddings.pkl', 'rb') as f1:
        entity_embeddings = pickle.load(f1)

            match_kg = []
            entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])

            for kg_entity in question_kg:
            
                keyword_index = keyword_embeddings["keywords"].index(kg_entity)
                kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])

                cos_similarities = cosine_similarity_manual(entity_embeddings_emb, kg_entity_emb)[0]
                max_index = cos_similarities.argmax()

                # ?
                if cos_similarities.argmax() < 0.5:
                    continue
                        
                match_kg_i = entity_embeddings["entities"][max_index]
                while match_kg_i in match_kg:
                    cos_similarities[max_index] = 0
                    max_index = cos_similarities.argmax()
                    match_kg_i = entity_embeddings["entities"][max_index]

                match_kg.append(match_kg_i)
            print('match_kg',match_kg)

            