from neo4j import GraphDatabase
import pandas as pd


def check_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    print(f"Error: Invalid line {line_number}: {line.strip()}")
                    return False
        print("File check passed.")
        return True
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return False


def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, 1):
            parts = line.strip().split('\t')
            if len(parts) == 3:
                data.append(parts)
            else:
                print(f"Warning: Skipping invalid line {line_number}: {line.strip()}")
    df = pd.DataFrame(data, columns=['head', 'relation', 'tail'])
    return df


def create_knowledge_graph(uri, user, password, df):
    driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=200)
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")  # 清空图数据库

        for index, row in df.iterrows():
            head_name = row['head']
            tail_name = row['tail']
            relation_name = row['relation'].replace('`', '').replace("'", "").replace('"', '')  # 处理特殊字符

            query = (
                f"MERGE (h:Entity {{ name: $head_name }}) "
                f"MERGE (t:Entity {{ name: $tail_name }}) "
                f"MERGE (h)-[:`{relation_name}`]->(t)"
            )
            try:
                session.run(query, head_name=head_name, tail_name=tail_name)
            except Exception as e:
                print(f"Error executing query for row {index}: {e}")

    driver.close()


def test_connection(uri, user, password):
    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session() as session:
                result = session.run("RETURN 1 AS x")
                print("Connection successful!")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False


# Construct gmail INSTANCE1
if __name__ == "__main__":
    uri = "neo4j+s://???.databases.neo4j.io"
    username = "neo4j"
    password = "password"

    file_path = '/data1/yuanxujie/KGE-DST/data/knowledge_graph.txt'

    if test_connection(uri, username, password):
        if check_file(file_path):
            df = load_data(file_path)
            create_knowledge_graph(uri, username, password, df)
    else:
        print("Please check your Neo4j connection settings and try again.")