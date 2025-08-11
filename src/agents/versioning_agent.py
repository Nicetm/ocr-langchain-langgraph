# src/agents/versioning_agent.py
from typing import List, Dict, Any, Tuple
from datetime import datetime

def _to_dt(s: str):
    try:
        return datetime.strptime(s, "%d-%m-%Y")
    except Exception:
        return None

class VersioningAgent:
    """
    Asigna versiones por categoría:
      input:  [{filename, clasificacion, fecha}, ...]
      output: {
        "constitucion": [{"filename":..., "fecha":"dd-mm-aaaa", "version":1}, ...],
        "inscripcion": [...],
        "publicacion": [...],
        "cedula": [...],
        "otros": [...]
      }
    """

    def assign_versions(self, classification_results: List[Dict[str, Any]],
                        date_results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        # mapa rápido de fechas por archivo (por si quieres corroborar más de una)
        fechas_map: Dict[str, Dict[str, str]] = {
            d["filename"]: {
                "fecha_mas_antigua": d.get("fecha_mas_antigua", ""),
                "fecha_mas_reciente": d.get("fecha_mas_reciente", "")
            }
            for d in date_results
        }

        # 1) agrupar por categoría
        grupos: Dict[str, List[Dict[str, Any]]] = {}
        for item in classification_results:
            cat = item.get("clasificacion", "otros")
            fn = item.get("filename")
            # preferimos fecha_mas_antigua del extractor
            fm = fechas_map.get(fn, {})
            fecha = fm.get("fecha_mas_antigua") or item.get("fecha") or ""

            grupos.setdefault(cat, []).append({
                "filename": fn,
                "fecha": fecha
            })

        # 2) ordenar por fecha asc y versionar
        versionado: Dict[str, List[Dict[str, Any]]] = {}
        for cat, docs in grupos.items():
            # split por con-fecha / sin-fecha
            con_fecha = [d for d in docs if _to_dt(d["fecha"])]
            sin_fecha = [d for d in docs if not _to_dt(d["fecha"])]

            con_fecha.sort(key=lambda d: (_to_dt(d["fecha"]), d["filename"]))
            # sin fecha al final, orden alfabético para ser determinista
            sin_fecha.sort(key=lambda d: d["filename"])

            ordenados = con_fecha + sin_fecha

            # asignar versiones
            out_cat = []
            for idx, d in enumerate(ordenados, start=1):
                out_cat.append({
                    "filename": d["filename"],
                    "fecha": d["fecha"],
                    "version": idx
                })
            versionado[cat] = out_cat

        return versionado

    def assign_versions_from_ocr(self, ocr_results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Clasifica documentos por categoría usando LLM, extrae fechas del texto OCR,
        y asigna versiones basándose en la fecha real del documento.
        """
        from src.config import get_llm
        from langchain.prompts import ChatPromptTemplate
        from langchain.schema.output_parser import StrOutputParser
        import re
        from datetime import datetime
        
        llm = get_llm()
        
        # Prompt para clasificación
        classification_prompt = ChatPromptTemplate.from_template("""
Eres un experto en documentos legales chilenos. Clasifica el documento en UNA de estas categorías según su contenido:

- escritura: Documentos de constitución o modificación de empresas (escrituras públicas, estatutos, actas de directorio, etc.)
- publicacion: Publicaciones en Diario Oficial o avisos legales
- inscripcion: Documentos de registro oficial (inscripciones, certificados de vigencia, etc.)
- cedula: Documentos de identificación personal

Responde EXCLUSIVAMENTE con una palabra: escritura | publicacion | inscripcion | cedula

TEXTO:
{text}
""")
        
        # Prompt para extracción de fecha
        date_prompt = ChatPromptTemplate.from_template("""
Extrae la fecha principal del documento legal. Busca fechas como:
- "En ciudad xxx a 20 de mes xxxx de 2021"
- "Santiago, 15 de enero de 2020"
- "Fecha: 10-03-2019"

Si encuentras múltiples fechas, usa la más relevante (generalmente la fecha de firma o constitución).

Responde ÚNICAMENTE en formato dd-mm-yyyy. Si no encuentras fecha, responde "NO_FECHA".

TEXTO:
{text}
""")
        
        classification_chain = classification_prompt | llm | StrOutputParser()
        date_chain = date_prompt | llm | StrOutputParser()
        
        # Procesar cada documento
        documentos_procesados = []
        
        for item in ocr_results:
            filename = item.get("filename", "")
            text = item.get("text", "")
            
            try:
                # Clasificar documento
                categoria = classification_chain.invoke({"text": text}).strip().lower()
                
                # Extraer fecha
                fecha_texto = date_chain.invoke({"text": text}).strip()
                
                # Validar formato de fecha
                fecha_final = "NO_FECHA"
                if fecha_texto != "NO_FECHA":
                    # Intentar parsear la fecha
                    try:
                        # Buscar patrón dd-mm-yyyy
                        fecha_match = re.search(r'(\d{1,2})-(\d{1,2})-(\d{4})', fecha_texto)
                        if fecha_match:
                            dia, mes, año = fecha_match.groups()
                            fecha_final = f"{int(dia):02d}-{int(mes):02d}-{año}"
                        else:
                            # Intentar otros formatos
                            fecha_final = fecha_texto
                    except:
                        fecha_final = "NO_FECHA"
                
                documentos_procesados.append({
                    "filename": filename,
                    "categoria": categoria,
                    "fecha": fecha_final,
                    "text": text
                })
                
            except Exception as e:
                print(f"Error procesando {filename}: {e}")
                documentos_procesados.append({
                    "filename": filename,
                    "categoria": "otros",
                    "fecha": "NO_FECHA",
                    "text": text
                })
        
        # Agrupar por categoría
        grupos: Dict[str, List[Dict[str, Any]]] = {}
        for doc in documentos_procesados:
            cat = doc["categoria"]
            grupos.setdefault(cat, []).append(doc)
        
        # Ordenar por fecha y asignar versiones
        versionado: Dict[str, List[Dict[str, Any]]] = {}
        
        for categoria, docs in grupos.items():
            # Separar documentos con fecha y sin fecha
            con_fecha = [d for d in docs if d["fecha"] != "NO_FECHA"]
            sin_fecha = [d for d in docs if d["fecha"] == "NO_FECHA"]
            
            # Ordenar por fecha (más antigua primero)
            def parse_fecha(fecha_str):
                try:
                    return datetime.strptime(fecha_str, "%d-%m-%Y")
                except:
                    return datetime.max
            
            con_fecha.sort(key=lambda d: parse_fecha(d["fecha"]))
            sin_fecha.sort(key=lambda d: d["filename"])  # Sin fecha, orden alfabético
            
            # Combinar: primero los con fecha, luego los sin fecha
            ordenados = con_fecha + sin_fecha
            
            # Asignar versiones
            out_cat = []
            for idx, d in enumerate(ordenados, start=1):
                out_cat.append({
                    "filename": d["filename"],
                    "fecha": d["fecha"],
                    "version": idx
                })
            versionado[categoria] = out_cat
        
        return versionado

    def mostrar_versionado(self, versioning_results: Dict[str, List[Dict[str, Any]]]) -> None:
        # Imprime con el formato que acordamos
        for cat, docs in versioning_results.items():
            if not docs:
                continue
            print(f"documentos de {cat}")
            for d in docs:
                f = d.get("fecha") or ""
                v = d.get("version")
                print(f"{d['filename']} (fecha {f}) --> version {v}")
            print("")
