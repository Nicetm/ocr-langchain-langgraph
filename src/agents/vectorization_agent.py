# src/agents/vectorization_agent.py
import hashlib
from typing import List, Dict, Any, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Usa tu configuración centralizada
from src.config import get_vectorstore, get_psycopg2_connection_string


def _sha1(s: str) -> str:
    h = hashlib.sha1()
    h.update(s.encode("utf-8", errors="ignore"))
    return h.hexdigest()


class VectorizationAgent:
    """
    Vectoriza chunks en pgvector usando langchain_postgres.PGVector (desde src.config.get_vectorstore).

    - Inserta con IDs estables por (filename, version, chunk_index) para idempotencia.
    - Metadatos ricos: filename, version, fecha, clasificacion, razon_social, tipo_empresa, rut, collection, stable_id.
    """

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", " "],
            keep_separator=False,
        )

    def _build_docs(
        self,
        items: List[Dict[str, Any]],
        razon_social: str,
        tipo_empresa: str,
        rut: str,
        collection_name: str,
    ) -> Tuple[List[Document], int, List[str]]:
        """
        items: [{filename, text, fecha, clasificacion, version}, ...]
        Devuelve: (docs, total_chunks, ids)
        """
        docs: List[Document] = []
        ids: List[str] = []
        total_chunks = 0

        for it in items:
            raw_text = (it.get("text") or "").strip()
            if not raw_text:
                continue

            splits = self.splitter.split_text(raw_text)

            for idx, chunk in enumerate(splits):
                stable_id = _sha1(f"{it['filename']}|{it.get('version','')}|{idx}")

                meta = {
                    "collection": collection_name,
                    "filename": it["filename"],
                    "version": it.get("version"),
                    "fecha": it.get("fecha", ""),
                    "clasificacion": it.get("clasificacion", ""),
                    "razon_social": razon_social or "",
                    "tipo_empresa": tipo_empresa or "",
                    "rut": rut or "",
                    "stable_id": stable_id,  # opcional: útil para trazabilidad
                }

                docs.append(Document(page_content=chunk, metadata=meta))
                ids.append(stable_id)
                total_chunks += 1

        return docs, total_chunks, ids

    def _delete_by_ids(self, collection_name: str, ids: List[str]) -> None:
        """
        Borrado idempotente por 'id' en langchain_pg_embedding.
        Compatible con distintos esquemas de langchain_postgres (evita depender de metadata/cmetadata).
        """
        if not ids:
            return
        try:
            import psycopg2
            conn = psycopg2.connect(get_psycopg2_connection_string())
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM langchain_pg_embedding
                    WHERE collection_id = (SELECT uuid FROM langchain_pg_collection WHERE name = %s)
                      AND id = ANY(%s);
                    """,
                    (collection_name, ids),
                )
            conn.close()
        except Exception as e:
            print(f"[vectorstore] Aviso: no se pudo borrar por id: {e}")

    def process_documents(
        self,
        items: List[Dict[str, Any]],
        razon_social: str,
        tipo_empresa: str,
        rut: str,
        collection_name: str,
    ) -> Dict[str, Any]:
        """
        items esperado:
          [{ filename, text, fecha, clasificacion, version }, ...]
        """
        # 1) Construir documentos/chunks + ids estables
        docs, total_chunks, ids = self._build_docs(
            items=items,
            razon_social=razon_social,
            tipo_empresa=tipo_empresa,
            rut=rut,
            collection_name=collection_name,
        )

        if not docs:
            return {
                "collection": collection_name,
                "documentos_procesados": 0,
                "total_chunks": 0,
                "status": "sin_texto",
            }

        # 2) Borrado idempotente por id
        self._delete_by_ids(collection_name, ids)

        # 3) Vector store desde tu config
        vs = get_vectorstore(collection_name=collection_name)

        # 4) Insertar con ids (para mantener idempotencia entre corridas)
        vs.add_documents(docs, ids=ids)

        return {
            "collection": collection_name,
            "documentos_procesados": len({d.metadata["filename"] for d in docs}),
            "total_chunks": total_chunks,
            "status": "ok",
        }
