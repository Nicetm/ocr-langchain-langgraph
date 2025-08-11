# src/agents/version_comparer_agent.py
from typing import Dict, Any, List, Optional
import json
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# ---------- Prompts ----------

COMPARE_PROMPT = ChatPromptTemplate.from_template("""
Eres un experto en documentos legales chilenos. Compara dos versiones de un documento y encuentra las diferencias principales.

Responde ÚNICAMENTE en formato JSON con esta estructura:
{{
  "cambios": [
    {{
      "titulo": "Descripción del cambio",
      "antes": "Texto que estaba en la versión anterior",
      "despues": "Texto que está en la versión nueva"
    }}
  ],
  "resumen": "Resumen breve de los cambios principales"
}}

Instrucciones:
- Encuentra las diferencias más importantes entre las dos versiones
- No te enfoques en campos específicos, busca cualquier cambio relevante
- Sé claro y conciso en las descripciones
- Si no hay cambios significativos, devuelve una lista vacía de cambios

VERSIÓN ANTERIOR:
----------------
{texto_anterior}
----------------

VERSIÓN NUEVA:
----------------
{texto_nueva}
----------------
""".strip())

class VersionComparerAgent:
    """
    Compara versiones v2..vN contra v1 por categoría.
    - Usa resúmenes dirigidos para reducir ruido.
    - Devuelve JSON de cambios por categoría.
    """

    def __init__(self, get_llm_callable=None, max_chars_doc: int = 16000):
        if get_llm_callable:
            self.llm = get_llm_callable()
        else:
            from src.config import get_llm
            self.llm = get_llm()
        self.max_chars_doc = max_chars_doc
        self.cmp_prompt = COMPARE_PROMPT
        self.parser = StrOutputParser()

    # -------- utilidades --------
    def _find_text(self, ocr_index: Dict[str, str], filename: str) -> str:
        return (ocr_index.get(filename) or "")[: self.max_chars_doc]

    def _compare_texts(self, texto_anterior: str, texto_nueva: str) -> Dict[str, Any]:
        """Compara directamente dos textos y encuentra las diferencias."""
        if not texto_anterior or not texto_nueva:
            return {"cambios": [], "resumen": ""}
        
        try:
            msgs = self.cmp_prompt.format_messages(
                texto_anterior=texto_anterior[:8000],  # Limitar tamaño para evitar tokens excesivos
                texto_nueva=texto_nueva[:8000]
            )
            raw = self.llm.invoke(msgs)
            if hasattr(raw, "content"):
                raw = raw.content
            try:
                data = json.loads(raw)
                if isinstance(data, dict) and "cambios" in data:
                    return data
            except Exception:
                pass
        except Exception:
            pass
        # fallback
        return {"cambios": [], "resumen": ""}

    # -------- API principal --------
    def comparar_versiones(
        self,
        versioning_results: Dict[str, List[Dict[str, Any]]],
        ocr_results: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        versioning_results: {categoria: [{filename, fecha, version}, ...], ...}
        ocr_results: [{filename, text, ...}]

        Devuelve:
        {
          "escritura": [
            {
              "categoria": "escritura",
              "de": {"filename": "...", "fecha": "dd-mm-aaaa", "version": 1},
              "a":  {"filename": "...", "fecha": "dd-mm-aaaa", "version": 2},
              "cambios": [...],
              "resumen": "..."
            },
            ...
          ],
          "publicacion": [...],
          ...
        }
        """
        # index de texto OCR por filename
        ocr_index = {d.get("filename"): d.get("text") or "" for d in ocr_results}

        resultados: Dict[str, List[Dict[str, Any]]] = {}

        for categoria, docs in versioning_results.items():
            if not docs or len(docs) < 2:
                resultados[categoria] = []
                continue

            # Ordenar por versión
            docs_ordenados = sorted(docs, key=lambda x: x.get("version", 0))
            
            comps: List[Dict[str, Any]] = []
            
            # Comparar versiones consecutivas
            for i in range(len(docs_ordenados) - 1):
                version_actual = docs_ordenados[i]
                version_siguiente = docs_ordenados[i + 1]
                
                texto_actual = self._find_text(ocr_index, version_actual["filename"])
                texto_siguiente = self._find_text(ocr_index, version_siguiente["filename"])
                
                diffs = self._compare_texts(texto_actual, texto_siguiente)

                comps.append({
                    "de": {"filename": version_actual["filename"], "fecha": version_actual.get("fecha",""), "version": version_actual.get("version")},
                    "a":  {"filename": version_siguiente["filename"], "fecha": version_siguiente.get("fecha",""), "version": version_siguiente.get("version")},
                    "diferencias": diffs
                })

            resultados[categoria] = comps

        return resultados

    def mostrar_comparaciones(self, comparison_results: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Imprime en consola de forma legible (como tus ejemplos).
        """
        for categoria, comps in comparison_results.items():
            if not comps:
                continue
            print(f"\n{categoria.upper()}:")
            for c in comps:
                de = c["de"]; a = c["a"]
                print(f"De {de['filename']} (fecha {de.get('fecha','')}) --> versión {de['version']}")
                print(f"A {a['filename']} (fecha {a.get('fecha','')}) --> versión {a['version']}")
                print("MODIFICACIONES:")

                diffs = c.get("diferencias", {})
                cambios = diffs.get("cambios", [])
                if not cambios:
                    print("  (Sin cambios significativos detectados)")
                else:
                    for cambio in cambios:
                        titulo = cambio.get("titulo", "Cambio")
                        antes = cambio.get("antes","")
                        despues = cambio.get("despues","")
                        print(f"  - {titulo}")
                        if antes and despues:
                            print(f"    Antes: {antes}")
                            print(f"    Después: {despues}")
                
                resumen = diffs.get("resumen","")
                if resumen:
                    print(f"  Resumen: {resumen}")
                print("")
