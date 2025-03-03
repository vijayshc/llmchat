import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import asyncio  # Import asyncio

AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT", "https://models.inference.ai.azure.com")
AZURE_API_TOKEN = os.environ.get("AZURE_API_TOKEN", "xx")
MODEL_NAME = os.environ.get("MODEL_NAME", "Meta-Llama-3.1-8B-Instruct")

def load_schema(schema_path='schema.json'):
    # Load the database schema from a JSON file
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    return schema

def get_column_list(schema):
    # Prepare column list in the format "table_name.column_name"
    column_list = []
    for table in schema['tables']:
        for column in table['columns']:
            column_list.append(f"{table['name']}.{column}")
    return column_list

def format_schema_for_llm(schema, relevant_tables=None):
    """Format the schema in a way that's easy for the LLM to understand"""
    formatted_schema = []
    
    tables = schema['tables']
    if relevant_tables:
        tables = [table for table in tables if table['name'] in relevant_tables]
    
    for table in tables:
        table_str = f"Table: {table['name']}\nColumns:\n"
        for column in table['columns']:
            table_str += f"- {column}"
            if 'type' in table:
                table_str += f" ({table['type']})"
            table_str += "\n"
        
        if 'foreign_keys' in table:
            table_str += "Foreign Keys:\n"
            for fk in table['foreign_keys']:
                table_str += f"- {fk['column']} references {fk['reference_table']}.{fk['reference_column']}\n"
                
        formatted_schema.append(table_str)
    
    return "\n".join(formatted_schema)

def get_few_shot_examples():
    """Provide few-shot examples to guide the LLM"""
    return [
        {"query": "Find all employees in the IT department",
         "sql": "SELECT * FROM employees WHERE department = 'IT'"},
        {"query": "Show me the total sales for each product category",
         "sql": "SELECT category, SUM(sales_amount) as total_sales FROM sales JOIN products ON sales.product_id = products.id GROUP BY category"},
        {"query": "List customers who made a purchase last month",
         "sql": "SELECT DISTINCT c.* FROM customers c JOIN orders o ON c.id = o.customer_id WHERE o.order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH)"}
    ]

def generate_sql_with_api(schema_string, query):
    """Generate SQL using the Azure AI Inference client for GitHub Models API"""
    # Format examples
    few_shot = get_few_shot_examples()
    few_shot_text = "\n\n".join([f"Query: {ex['query']}\nSQL: {ex['sql']}" for ex in few_shot])
    
    # Construct the prompt
    prompt = f"""You are an expert SQL query generator. Given a database schema and a natural language query, 
generate the correct SQL query that answers the question.

DATABASE SCHEMA:
{schema_string}

EXAMPLES:
{few_shot_text}

Now generate SQL for the following query:
Query: {query}
SQL:"""
    print("Prompt:"+prompt)
    print(f"Generating SQL with Azure AI Inference client for GitHub Models API...")
    
    # Initialize the client
    client = ChatCompletionsClient(
        endpoint=AZURE_ENDPOINT,
        credential=AzureKeyCredential(AZURE_API_TOKEN),
    )
    
    # Send the request to the API
    response = client.complete(
        stream=True,
        messages=[
            SystemMessage("You are a helpful assistant."),
            UserMessage(prompt),
        ],
        model_extras={'stream_options': {'include_usage': True}},
        model=MODEL_NAME,
    )
    
    sql_result = ""
    usage = {}
    for update in response:
        if update.choices and update.choices[0].delta:
            sql_result += update.choices[0].delta.content or ""
        if update.usage:
            usage = update.usage

    if usage:
        print("\nUsage:")
        for k, v in usage.items():
            print(f"{k} = {v}")

    client.close()
    
    # Extract just the SQL part from the result
    if "SQL:" in sql_result:
        sql_result = sql_result.split("SQL:")[-1].strip()
    
    return sql_result

# NEW: Asynchronous function to embed text
async def embed_text(text, embedder):
    return embedder.encode(text)

# NEW: Function to compute table relevance using aggregated table text
async def get_relevant_tables(schema, query, embedder, top_m=3):
    table_embeddings = []
    table_names = []
    
    # NEW: Group tables into clusters based on business_function
    table_clusters = {}
    for table in schema['tables']:
        cluster_name = table.get('business_function', 'default')  # Use business_function from schema
        if cluster_name not in table_clusters:
            table_clusters[cluster_name] = []
        table_clusters[cluster_name].append(table)
    
    # NEW: First, find relevant table clusters based on the query
    cluster_embeddings = {}
    for cluster_name, tables in table_clusters.items():
        cluster_text = " ".join([table['name'] for table in tables])
        cluster_embeddings[cluster_name] = await embed_text(cluster_text, embedder)
    
    # NEW: Compute similarities between query and cluster embeddings
    cluster_similarities = {}
    query_emb = await embed_text(query, embedder)
    for cluster_name, cluster_emb in cluster_embeddings.items():
        cluster_emb = np.array(cluster_emb)
        query_emb = np.array(query_emb)
        norm_cluster = cluster_emb / np.linalg.norm(cluster_emb)
        norm_query = query_emb / np.linalg.norm(query_emb)
        cluster_similarities[cluster_name] = np.dot(norm_cluster, norm_query)
    
    # NEW: Select top clusters
    top_cluster_names = sorted(cluster_similarities, key=cluster_similarities.get, reverse=True)[:2]  # Select top 2 clusters
    relevant_tables = []
    for cluster_name in top_cluster_names:
        relevant_tables.extend(table_clusters[cluster_name])
    
    # NEW: Compute embeddings only for tables in the relevant clusters
    for table in relevant_tables:
        table_text = table['name'] + " " + " ".join(table['columns'])
        table_emb = await embed_text(table_text, embedder)
        table_embeddings.append(table_emb)
        table_names.append(table['name'])
    
    table_embeddings = np.array(table_embeddings)
    
    # NEW: Compute similarities between query and table embeddings
    norm_table = table_embeddings / np.linalg.norm(table_embeddings, axis=1, keepdims=True)
    norm_query = query_emb / np.linalg.norm(query_emb)
    similarities = np.dot(norm_table, norm_query)
    top_indices = similarities.argsort()[-top_m:][::-1]
    
    return [table_names[i] for i in top_indices]

async def main():
    try:
        # Load schema
        print("Loading schema...")
        schema = load_schema()

        # Define the natural language query
        query = "Show me the names of all employees who joined after 2010 but before 2025 along with their project and department details"

        # Initialize the sentence transformer for embeddings
        print("Loading sentence transformer model...")
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # OPTIMIZATION: Use aggregated table embeddings instead of column-level FAISS indexing
        top_m_tables = await get_relevant_tables(schema, query, embedder, top_m=3)
        print(f"Most relevant tables: {top_m_tables}")
        
        # Format schema for the selected tables
        schema_string = format_schema_for_llm(schema, top_m_tables)
        
        try:
            # Generate SQL using the Azure AI Inference client for GitHub Models API
            sql_query = generate_sql_with_api(schema_string, query)
            
            print("\nGenerated SQL Query:")
            print("--------------------")
            print(sql_query)
            
        except Exception as e:
            print(f"Error with Azure AI Inference client for GitHub Models API: {str(e)}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
