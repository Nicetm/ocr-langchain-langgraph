"""
Configuración simple para el proyecto.
"""

import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai.chat_models import AzureChatOpenAI

load_dotenv()

# Configuración de PostgreSQL
PG_HOST = os.getenv('PG_HOST', 'localhost')
PG_PORT = os.getenv('PG_PORT', '5432')
PG_DATABASE = os.getenv('PG_DATABASE', 'postgres')
PG_USER = os.getenv('PG_USER', 'postgres')
PG_PASSWORD = os.getenv('PG_PASSWORD', '123456')
PG_CONN_STR = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
PG_PSYCOPG2_STR = f"host={PG_HOST} port={PG_PORT} dbname={PG_DATABASE} user={PG_USER} password={PG_PASSWORD}"

def get_embeddings():
    """Retorna embeddings configurados."""
    return AzureOpenAIEmbeddings(
        model=os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION')
    )

def get_vectorstore(collection_name="default"):
    """Retorna vectorstore con similitud coseno."""
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=collection_name,
        connection=PG_CONN_STR,
        distance_strategy="cosine",
        use_jsonb=True
    )

def get_llm():
    """Retorna LLM configurado."""
    return AzureChatOpenAI(
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
        temperature=0
    )

def get_psycopg2_connection_string():
    """Retorna cadena de conexión para psycopg2."""
    return PG_PSYCOPG2_STR

def should_extract_from_ocr():
    """Retorna True si se debe extraer información desde OCR directamente."""
    return os.getenv('EXTRACT_FROM_OCR', 'true').lower() == 'true' 