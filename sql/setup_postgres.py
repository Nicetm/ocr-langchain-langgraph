"""
Script para configurar PostgreSQL con la extensión pgvector.
"""

import os
import psycopg2
from dotenv import load_dotenv

def setup_postgres():
    """Configura PostgreSQL con la extensión pgvector."""
    load_dotenv()
    
    # Obtener configuración de PostgreSQL
    pg_host = os.getenv("PG_HOST", "localhost")
    pg_port = os.getenv("PG_PORT", "5432")
    pg_database = os.getenv("PG_DATABASE", "postgres")
    pg_user = os.getenv("PG_USER", "postgres")
    pg_password = os.getenv("PG_PASSWORD", "123456")
    
    try:
        # Conectar a PostgreSQL
        conn = psycopg2.connect(
            host=pg_host,
            port=pg_port,
            database=pg_database,
            user=pg_user,
            password=pg_password
        )
        
        cursor = conn.cursor()
        
        # Verificar si la extensión pgvector está disponible
        cursor.execute("SELECT * FROM pg_available_extensions WHERE name = 'vector';")
        if not cursor.fetchone():
            print("ERROR: La extensión pgvector no está disponible en PostgreSQL.")
            print("Instala pgvector en tu servidor PostgreSQL.")
            return False
        
        # Crear la extensión vector si no existe
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        
        print("Extensión pgvector habilitada correctamente.")
        
        # Verificar que la extensión está activa
        cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        if cursor.fetchone():
            print("Extensión vector está activa.")
        else:
            print("Error: La extensión vector no se pudo activar.")
            return False
        
        cursor.close()
        conn.close()
        
        print("PostgreSQL configurado correctamente para vectorización.")
        return True
        
    except Exception as e:
        print(f"Error configurando PostgreSQL: {e}")
        print("\nAsegúrate de que:")
        print("1. PostgreSQL esté ejecutándose")
        print("2. Las credenciales en .env sean correctas")
        print("3. La extensión pgvector esté instalada")
        return False

if __name__ == "__main__":
    setup_postgres() 