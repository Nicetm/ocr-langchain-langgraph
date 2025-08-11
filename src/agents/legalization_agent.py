from typing import List, Dict, Any, Optional, Callable, Tuple
import json
import re
import os
import psycopg2
import psycopg2.extras
from dataclasses import dataclass
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from src.config import get_psycopg2_connection_string, get_vectorstore



# ========================
# PROMPT DE VERIFICACIÓN
# ========================

VERIFY_PROMPT = ChatPromptTemplate.from_template("""
Eres experto en documentos legales chilenos. Verifica si el siguiente fragmento de texto
OTORGA (menciona de forma expresa) el poder descrito por un CÓDIGO de catálogo.

Responde SOLO en JSON con esta estructura:
{{
  "otorgado": true/false,
  "actor": "quién queda facultado (si se indica; ej.: Administrador estatutario, Directorio, socio, representante legal, etc.)",
  "limites": "límites monetarios o de actuación, si los hay (ej.: 500 UF individual, conjunta sin tope)",
  "restricciones": "condiciones explícitas (ej.: requiere Junta Extraordinaria, requiere acuerdo de accionistas, etc.)",
  "evidencia": "cita textual corta que sustente la decisión",
  "confianza": "alta|media|baja",
  "motivo_no_otorgado": "si es false, explica brevemente por qué (p.ej.: negación expresa, mera referencia, no aparece)"
}}

INSTRUCCIONES:
- "Otorgado" solo si el texto contiene mención EXPRESA que habilite el poder.
- Rechaza si es mera mención genérica o histórica.
- Aplica sentido legal: si el texto dice "NO autoriza" o "requiere acuerdo previo" trátalo como no otorgado o con restricción.
- No inventes. Usa solo el fragmento dado.

CÓDIGO:
- id: {codigo_id}
- nombre: {codigo_nombre}
- descripcion: {codigo_descripcion}
- palabras_claves: {codigo_palabras}

FRAGMENTO:
----------------
{fragmento}
----------------
""".strip())


# ========================
# CONTRATOS / UTILIDADES
# ========================

@dataclass
class LegalCode:
    id: str
    nombre: str
    descripcion: str
    palabras_claves: List[str]
    grupo: Optional[str] = None
    anclas_obligatorias: Optional[List[str]] = None  # p.ej. ["cuenta corriente","abrir","cerrar"] o ["cheque|cheques","endosar|endoso"]


def normalize_text(s: str) -> str:
    s = (s or "").replace("\xa0", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    # Normaliza abreviaturas frecuentes
    s = s.replace("ctas. ctes.", "cuentas corrientes")
    s = s.replace("cts. ctes.", "cuentas corrientes")
    s = s.replace("c/c", "cuenta corriente")
    return s


def split_chunks(text: str, max_len: int = 1000, overlap: int = 120) -> List[str]:
    text = normalize_text(text)
    if len(text) <= max_len:
        return [text]
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + max_len])
        i += max_len - overlap
    return chunks


def _anchor_variant_in_text(text_low: str, anchor_expr: str) -> bool:
    """
    Soporta alternativas con '|'. Ej: "cheque|cheques" → match si aparece "cheque" o "cheques".
    Haz match por substring simple (rápido y suficiente con normalización previa).
    """
    # Permite expresiones tipo "palabra1|palabra2"
    variants = [v.strip().lower() for v in anchor_expr.split("|") if v.strip()]
    if not variants:
        return False
    return any(v in text_low for v in variants)


def contains_all_anchors(text: str, anchors: List[str]) -> bool:
    """
    Devuelve True si para CADA ancla se cumple al menos una de sus variantes.
    Ej: ["cuenta corriente", "abrir|apertura", "cerrar|clausurar"].
    """
    if not anchors:
        return True
    tlow = (text or "").lower()
    for anchor in anchors:
        if not _anchor_variant_in_text(tlow, anchor):
            return False
    return True


# ========================
# AGENTE PRINCIPAL
# ========================

class LegalizationAgent:
    """
    Extrae 'poderes' (legalización) mapeándolos a un catálogo de códigos.
    - Usa retriever (opcional) para recall, reglas de anclas para precisión,
      y un LLM para verificación/extracción final.
    - Trabaja típicamente sobre la última versión de 'escritura'.
    """

    def __init__(
        self,
        get_llm_callable: Optional[Callable[[], Any]] = None,
        retriever_callable: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
        max_chars_doc: int = 16000,
        min_similarity: float = 0.70
    ):
        if get_llm_callable:
            self.llm = get_llm_callable()
        else:
            # Debe existir en tu proyecto
            from src.config import get_llm
            self.llm = get_llm()

        self.parser = StrOutputParser()
        self.verify_prompt = VERIFY_PROMPT

        # retriever(query, k)-> List[{text, score, metadata}]
        self.retriever = retriever_callable
        self.max_chars_doc = max_chars_doc
        self.min_similarity = min_similarity

    # ------------------
    # Carga catálogo BD
    # ------------------

    def cargar_catalogo_desde_bd(self) -> List[Dict[str, Any]]:
        """
        Lee langchain_pg_facultades y devuelve una lista de dicts:
        [
          {
            "id": "01",
            "nombre": "...",
            "descripcion": "...",
            "palabras_claves": "a, b, c",
            "grupo": "Cuentas Corrientes",
            "anclas_obligatorias": ["...", "..."]  # puede venir [] o None
          },
          ...
        ]
        """
        from src.config import should_extract_from_ocr
        
        # Si está en modo sin vectorización, no intentar conectar a PostgreSQL
        if not should_extract_from_ocr():
            print("[LEGALIZATION] Modo sin vectorización: omitiendo conexión a PostgreSQL para catálogo")
            return []
            
        conn = psycopg2.connect(get_psycopg2_connection_string())
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute("""
                    SELECT
                        codigo::text AS id,
                        nombre,
                        descripcion,
                        palabras_claves,
                        grupo,
                        anclas_obligatorias
                    FROM langchain_pg_facultades
                    ORDER BY
                        CASE WHEN codigo ~ '^[0-9]+$' THEN 0 ELSE 1 END,
                        codigo::int NULLS LAST,
                        codigo
                """)
                rows = cur.fetchall()

            catalogo = []
            for r in rows:
                anclas = r["anclas_obligatorias"] if r["anclas_obligatorias"] is not None else []
                catalogo.append({
                    "id": (r["id"] or "").strip(),
                    "nombre": (r["nombre"] or "").strip(),
                    "descripcion": (r["descripcion"] or "").strip(),
                    "palabras_claves": (r["palabras_claves"] or "").strip(),
                    "grupo": (r["grupo"] or "").strip(),
                    "anclas_obligatorias": anclas if isinstance(anclas, list) else [],
                })
            return catalogo
        finally:
            conn.close()

    # ------------------
    # Retriever PGVector (opcional)
    # ------------------

    def mi_retriever_vectorstore(self, collection: str, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Usa langchain_postgres.PGVector para recuperar top-k documentos similares.
        Requiere:
        - Variable de entorno PG_CONN: "postgresql+psycopg://user:pass@host:5432/db"
        - OPENAI_API_KEY para embeddings
        - Colección `collection` creada con los mismos embeddings al indexar.
        """

        store = get_vectorstore(collection_name=collection)
        
        retriever = store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)

        out = []
        for d in docs:
            # Algunos backends no incluyen similarity; dejamos 1.0 por defecto
            score = float(d.metadata.get("similarity", 1.0) or 1.0)
            out.append({
                "text": d.page_content,
                "score": score,
                "metadata": d.metadata
            })
        return out

    # ------------------
    # Helpers internos
    # ------------------

    def _candidate_chunks_for_code(
        self,
        code: LegalCode,
        full_text: str,
        filename: str = ""
    ) -> List[Tuple[str, float]]:
        """
        Devuelve (chunk, score_aprox).
        Si hay retriever, usa top-k desde el vectorstore; si no, hace chunks del texto completo (score=1.0).
        """
        if self.retriever:
            query = " ; ".join([code.nombre, code.descripcion, ", ".join(code.palabras_claves)])
            try:
                hits = self.retriever(query, 6)  # top-k
            except Exception:
                hits = []
            out: List[Tuple[str, float]] = []
            for h in hits:
                score = float(h.get("score", 1.0))
                # Si tu retriever no trae score real, no filtres por score (o deja min_similarity bajo)
                if score >= self.min_similarity:
                    out.append((normalize_text(h.get("text", ""))[:1000], score))
            # Si no hubo hits, hacer fallback a chunks completos
            if not out and full_text:
                chunks = split_chunks(full_text, 1000, 120)
                out = [(c, 1.0) for c in chunks]
            return out

        # Sin retriever: chunkear el documento completo
        chunks = split_chunks(full_text, 1000, 120)
        return [(c, 1.0) for c in chunks]

    def _rule_filter(self, code: LegalCode, chunk: str) -> bool:
        """
        Filtro por anclas obligatorias (si existen).
        Evita falsos positivos antes del LLM.
        """
        if code.anclas_obligatorias:
            return contains_all_anchors(chunk, code.anclas_obligatorias)
        return True

    def _verify_with_llm(self, code: LegalCode, chunk: str) -> Dict[str, Any]:
        msgs = self.verify_prompt.format_messages(
            codigo_id=code.id,
            codigo_nombre=code.nombre,
            codigo_descripcion=code.descripcion,
            codigo_palabras=", ".join(code.palabras_claves or []),
            fragmento=chunk[:4000]
        )
        raw = self.llm.invoke(msgs)
        content = raw.content if hasattr(raw, "content") else str(raw)
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "otorgado" in data:
                return data
        except Exception:
            pass
        # fallback seguro
        return {
            "otorgado": False,
            "actor": "",
            "limites": "",
            "restricciones": "",
            "evidencia": "",
            "confianza": "baja",
            "motivo_no_otorgado": "No se pudo interpretar el JSON del verificador"
        }

    # ------------------
    # API PÚBLICA
    # ------------------

    def extract_from_documents(
        self,
        ocr_docs: List[Dict[str, Any]],                       # [{filename, text, ...}]
        versioning_results: Dict[str, List[Dict[str, Any]]],  # {'escritura':[...], 'publicacion':[...], ...}
        code_catalog: List[Dict[str, Any]],                   # [{'id','nombre','descripcion','palabras_claves','grupo','anclas_obligatorias'?}, ...]
        only_latest_escritura: bool = True
    ) -> Dict[str, Any]:
        """
        Devuelve un JSON con poderes detectados:
        {
          "documento_base": {"filename","fecha","version"},
          "poderes": [
            {"id","nombre","grupo","otorgado","actor","limites","restricciones","evidencia","confianza","archivo","fecha","version"}
          ]
        }
        """
        # Index por filename desde OCR
        txt_idx = {d.get("filename"): (d.get("text") or "")[:self.max_chars_doc] for d in ocr_docs}

        # Tomamos la(s) escritura(s)
        escrituras = versioning_results.get("escritura", []) or []
        if not escrituras:
            return {"documento_base": None, "poderes": []}

        escrituras_ordenadas = sorted(escrituras, key=lambda x: x.get("version", 0))
        base_doc = escrituras_ordenadas[-1] if only_latest_escritura else escrituras_ordenadas[0]
        base_filename = base_doc.get("filename")
        base_text = txt_idx.get(base_filename, "")

        if not (base_text or "").strip():
            return {"documento_base": base_doc, "poderes": []}

        # Normaliza catálogo -> LegalCode
        codes: List[LegalCode] = []
        for row in code_catalog:
            codes.append(
                LegalCode(
                    id=str(row.get("id") or row.get("codigo") or "").strip(),
                    nombre=str(row.get("nombre") or "").strip(),
                    descripcion=str(row.get("descripcion") or "").strip(),
                    palabras_claves=[p.strip() for p in (row.get("palabras_claves") or "").split(",") if p.strip()],
                    grupo=row.get("grupo"),
                    anclas_obligatorias=row.get("anclas_obligatorias") or None
                )
            )

        poderes: List[Dict[str, Any]] = []

        # Para cada código, buscar candidatos y verificar
        for code in codes:
            candidates = self._candidate_chunks_for_code(code, base_text, base_filename)

            best: Optional[Dict[str, Any]] = None
            for chunk, score in candidates:
                if not self._rule_filter(code, chunk):
                    continue

                ver = self._verify_with_llm(code, chunk)

                if ver.get("otorgado") is True:
                    cand = {
                        "id": code.id,
                        "nombre": code.nombre,
                        "grupo": code.grupo,
                        "otorgado": True,
                        "actor": ver.get("actor", ""),
                        "limites": ver.get("limites", ""),
                        "restricciones": ver.get("restricciones", ""),
                        "evidencia": ver.get("evidencia", ""),
                        "confianza": ver.get("confianza", "media"),
                        "archivo": base_filename,
                        "fecha": base_doc.get("fecha", ""),
                        "version": base_doc.get("version", 0)
                    }
                    if (best is None) or (best.get("confianza") != "alta" and cand["confianza"] == "alta"):
                        best = cand
                    if best and best["confianza"] == "alta":
                        break  # ya suficiente

            # Solo agregar si está otorgado
            if best is not None:
                poderes.append(best)

        return {
            "documento_base": {
                "filename": base_filename,
                "fecha": base_doc.get("fecha", ""),
                "version": base_doc.get("version", 0)
            },
            "poderes": poderes
        }
