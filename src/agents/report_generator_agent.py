# src/agents/report_generator_agent.py
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from src.config import get_llm, get_psycopg2_connection_string

# =============== Utilidades ===============

def _to_dt(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(s, "%d-%m-%Y")
    except Exception:
        return None

def _select_latest_by_category(
    versioning_results: Dict[str, List[Dict[str, Any]]],
    category: str
) -> Optional[Dict[str, Any]]:
    """
    Devuelve el dict del documento más reciente (mayor 'version') para la categoría.
    Estructura esperada: {"filename":..., "fecha":"dd-mm-aaaa", "version":N}
    """
    docs = versioning_results.get(category) or []
    if not docs:
        return None
    # vienen ordenados por fecha asc y con version asignada 1..N
    return max(docs, key=lambda d: (d.get("version") or 0))

def _ocr_text_by_filename(ocr_results: List[Dict[str, Any]], filename: str) -> str:
    idx = {d.get("filename"): d.get("text") or "" for d in ocr_results}
    return idx.get(filename, "")

def _latest_texts_pool(
    versioning_results: Dict[str, List[Dict[str, Any]]],
    ocr_results: List[Dict[str, Any]],
    top_per_cat: int = 2
) -> List[Tuple[str, str, str]]:
    """
    Construye un pool de textos recientes por categoría: [(cat, filename, text), ...]
    Toma las últimas 'top_per_cat' versiones por categoría.
    """
    pool = []
    idx_text = {d.get("filename"): d.get("text") or "" for d in ocr_results}
    for cat, docs in versioning_results.items():
        if not docs:
            continue
        # tomar últimas N por version
        ordered = sorted(docs, key=lambda d: d.get("version") or 0, reverse=True)
        for d in ordered[:top_per_cat]:
            pool.append((cat, d["filename"], idx_text.get(d["filename"], "")))
    return pool

# =============== Prompts ===============

ENCABEZADO_PROMPT = ChatPromptTemplate.from_template("""
Eres un asistente legal en Chile. Analiza el siguiente documento legal y extrae la información básica de la empresa.

IMPORTANTE: Responde ÚNICAMENTE en formato JSON válido con esta estructura exacta:
{{
  "informesociedad": {{
    "razon_social": "NOMBRE COMPLETO DE LA EMPRESA",
    "rut": "RUT DE LA EMPRESA (formato: XX.XXX.XXX-X)",
    "nombre_fantasia": "NOMBRE DE FANTASÍA (si existe, si no deja vacío)"
  }}
}}

INSTRUCCIONES DE EXTRACCIÓN:
- Busca la razón social en el encabezado, título o primera parte del documento
- El RUT debe incluir puntos y guión (ej: 76.154.106-4)
- El nombre de fantasía es opcional, solo si aparece explícitamente
- Si no encuentras información clara, usa valores vacíos
- Busca en TODO el documento, no solo en el encabezado

TEXTO DEL DOCUMENTO:
-------
{text}
-------
""".strip())

# Campos de Constitución / Capital / Administración / Legalización.
# El extractor trabaja sobre texto reciente. Si un campo no aparece, deja "" o [] según corresponda.
SECCIONES_PROMPT = ChatPromptTemplate.from_template("""
Eres abogado corporativo en Chile. Del texto siguiente, extrae SOLO los campos de la sección indicada.
Formato de salida OBLIGATORIO: JSON con los campos exactos (si no aparece, usa "" o []).

Sección: {seccion}

Campos por sección:

- constitucion:
{{
  "razon_social_anterior": "",
  "tipo_de_sociedad": "",
  "domicilio": "",             # si hay múltiples domicilios separa con " | "
  "objeto_social_resumen": "",
  "fecha_constitucion": "",    # dd-mm-aaaa (toma de las versiones más antiguas cuando aplique)
  "fecha_termino": "",
  "frecuencia_prorroga": "",
  "en_caso_de_fallecimiento": ""
}}

- capital_social:
{{
  "capital_suscrito": "",      # CLP si aparece
  "capital_pagado": "",
  "plazo_para_enterarlo": "",
  "cierre_ejercicio": "",
  "responsabilidad_socios": "",
  "distribucion_utilidades": "",
  "numero_acciones": "",
  "observaciones": ""
}}

- administracion:
{{
  "tipo_administracion": "",                # ej. unipersonal, directorio, administración conjunta, etc.
  "administradores_titulares_num": "",      # número
  "administradores_suplentes_num": "",      # número
  "duracion": "",
  "firmas_requeridas_num": "",              # número de firmas requeridas
  "representantes_legales_num": "",         # número
  "forma_de_actuar": "",
  "observaciones": ""                        # resumen corto
}}

- legalizacion:
{{
  "codigo_escritura": "",
  "tipo_escritura": "",
  "digitador": "",
  "fecha_ingreso": "",
  "tipo_documento": "",
  "repertorio": "",
  "notaria": "",
  "ciudad_notaria": "",
  "fecha_notaria": "",
  "fecha_publicacion_diario_oficial": "",
  "vigencia_desde": "",
  "vigencia_hasta": "",
  "inscripcion_registro_comercio": "",
  "observacion_inscripcion": ""
}}

INSTRUCCIONES IMPORTANTES:
- Busca información en TODO el texto, no solo en el encabezado
- Para domicilio: busca direcciones, ciudades, comunas, regiones
- Para capital: busca montos, valores, capital suscrito/pagado, acciones
- Para administración: busca nombres de administradores, tipo de administración, facultades
- Para legalización: busca datos de notaría, repertorio, inscripciones, fechas
- Si encuentras información parcial, inclúyela
- No dejes campos vacíos si hay información disponible en el texto
- Extrae fechas en formato dd-mm-aaaa cuando sea posible

TEXTO:
----------------
{text}
----------------
""".strip())

PODERES_PROMPT = ChatPromptTemplate.from_template("""
Eres abogado corporativo. Debes identificar ÚNICAMENTE los códigos de la siguiente tabla que estén EXPRESAMENTE mencionados en el texto.
Si NO hay mención expresa para un código, NO lo incluyas.

Tabla (grupo, codigo, nombre, descripcion, palabras_claves):
{tabla}

Responde EXCLUSIVAMENTE un JSON con esta forma:
{{
  "poderes_y_personerias": [
    {{
      "grupo": "<grupo>",
      "codigo": "<codigo>",
      "nombre": "<nombre>",
      "descripcion": "<descripcion>"
    }}
  ]
}}

TEXTO A ANALIZAR (texto reciente de escrituras/constitución/modificaciones, o inscripciones/publicaciones si no hay):
----------------
{text}
----------------
""".strip())

RESTRICCIONES_PROMPT = ChatPromptTemplate.from_template("""
Eres abogado corporativo en Chile. Del texto reciente (últimas versiones) identifica RESTRICCIONES explícitas aplicables a facultades o poderes.
Responde EXCLUSIVAMENTE un JSON:

{{
  "restricciones": [
    {{
      "descripcion": "<texto corto>",
      "facultades_afectadas": ["<codigo>", "<codigo>"]  # si se puede mapear a códigos conocidos, inclúyelos; si no, deja []
    }}
  ]
}}

TEXTO:
----------------
{text}
----------------
""".strip())

class ReportGeneratorAgent:
    """
    Arma el informe final con:
      - Encabezado
      - Sección Constitución / Capital Social / Administración / Legalización
      - Poderes y Personerías (cruzado con tabla langchain_pg_facultades)
      - Restricciones (de últimas versiones)
    """

    def __init__(self):
        self.llm = get_llm()
        self.parser = StrOutputParser()

    # -------------- DB: facultades --------------
    def _fetch_facultades(self) -> List[Dict[str,str]]:
        """
        Devuelve lista [{grupo,codigo,nombre,descripcion,palabras_claves}, ...]
        desde la tabla langchain_pg_facultades.
        """
        from src.config import should_extract_from_ocr
        
        # Si está en modo sin vectorización, no intentar conectar a PostgreSQL
        if not should_extract_from_ocr():
            print("[REPORT] Modo sin vectorización: omitiendo conexión a PostgreSQL para facultades")
            return []
            
        try:
            import psycopg2
            conn = psycopg2.connect(get_psycopg2_connection_string())
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT grupo, codigo, nombre, descripcion, palabras_claves
                    FROM langchain_pg_facultades
                    ORDER BY grupo, codigo::int
                """)
                rows = cur.fetchall()
            conn.close()
            out = []
            for g,c,n,d,p in rows:
                out.append({
                    "grupo": g, "codigo": c, "nombre": n,
                    "descripcion": d, "palabras_claves": p or ""
                })
            return out
        except Exception as e:
            print(f"[report] No se pudo leer langchain_pg_facultades: {e}")
            return []

    # -------------- LLM helpers --------------
    def _llm_json(self, prompt_tpl: ChatPromptTemplate, **kwargs) -> Dict[str, Any]:
        msgs = prompt_tpl.format_messages(**kwargs)
        raw = self.llm.invoke(msgs)
        if hasattr(raw, "content"):
            raw = raw.content
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    # -------------- Extractores por sección --------------
    def _build_encabezado(self, entity_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Construye el encabezado usando entity_results si existe, 
        o crea uno básico si no hay información.
        """
        if entity_results:
            # Si hay entity_results, usarlos
            razon_social = entity_results.get("razon_social_actual", "")
            rut = entity_results.get("rut", "")
            nombre_fantasia = entity_results.get("nombre_fantasia", "")
            data = self._llm_json(ENCABEZADO_PROMPT,
                                  razon_social=razon_social,
                                  rut=rut,
                                  nombre_fantasia=nombre_fantasia)
            # fallback si LLM devuelve vacío
            if not data:
                data = {"informesociedad": {
                    "razon_social": razon_social,
                    "rut": rut,
                    "nombre_fantasia": nombre_fantasia
                }}
            return data
        else:
            # Si no hay entity_results, crear uno básico
            return {"informesociedad": {
                "razon_social": "",
                "rut": "",
                "nombre_fantasia": ""
            }}

    def _extract_encabezado_with_date_priority(
        self,
        versioning_results: Dict[str, List[Dict[str, Any]]],
        ocr_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extrae información del encabezado siguiendo prioridad de fechas.
        Empieza por el documento más reciente y va hacia atrás solo para campos faltantes.
        """
        # Obtener documentos de escritura ordenados por fecha (más reciente primero)
        escritura_docs = versioning_results.get("escritura", [])
        if not escritura_docs:
            print("[REPORT] No hay documentos de escritura para extraer encabezado")
            return {"informesociedad": {"razon_social": "", "rut": "", "nombre_fantasia": ""}}
        
        # Ordenar por fecha (más reciente primero)
        escritura_docs_sorted = sorted(
            escritura_docs, 
            key=lambda x: self._parse_fecha(x.get("fecha", "")), 
            reverse=True
        )
        
        print("[REPORT] Extrayendo encabezado con prioridad de fechas:")
        for i, doc in enumerate(escritura_docs_sorted):
            print(f"[REPORT]   {i+1}. {doc['filename']} ({doc['fecha']})")
        
        # Extraer texto del documento más reciente
        latest_doc = escritura_docs_sorted[0]
        latest_text = _ocr_text_by_filename(ocr_results, latest_doc["filename"])
        
        if not latest_text:
            print(f"[REPORT] No se encontró texto para {latest_doc['filename']}")
            return {"informesociedad": {"razon_social": "", "rut": "", "nombre_fantasia": ""}}
        
        # Extraer información del documento más reciente
        print(f"[REPORT] Extrayendo encabezado desde {latest_doc['filename']} ({latest_doc['fecha']})")
        result = self._llm_json(ENCABEZADO_PROMPT, text=latest_text) or {}
        
        # Obtener el encabezado del resultado
        encabezado = result.get("informesociedad", {})
        
        # Verificar campos vacíos y buscar en documentos anteriores
        empty_fields = self._get_empty_fields(encabezado)
        
        if empty_fields:
            print(f"[REPORT] Campos faltantes en {latest_doc['filename']}: {empty_fields}")
            
            # Buscar en documentos anteriores solo para campos faltantes
            for doc in escritura_docs_sorted[1:]:
                if not empty_fields:  # Si ya no hay campos vacíos, parar
                    break
                    
                doc_text = _ocr_text_by_filename(ocr_results, doc["filename"])
                if not doc_text:
                    continue
                
                print(f"[REPORT] Buscando campos faltantes en {doc['filename']} ({doc['fecha']})")
                doc_result = self._llm_json(ENCABEZADO_PROMPT, text=doc_text) or {}
                doc_encabezado = doc_result.get("informesociedad", {})
                
                # Llenar solo campos que estén vacíos
                for field in empty_fields[:]:  # Copiar lista para iterar
                    if doc_encabezado.get(field) and doc_encabezado[field].strip():
                        encabezado[field] = doc_encabezado[field]
                        empty_fields.remove(field)
                        print(f"[REPORT]   ✓ Campo '{field}' encontrado en {doc['filename']}")
                
                if not empty_fields:
                    print("[REPORT] Todos los campos del encabezado completados")
                    break
        
        return {"informesociedad": encabezado}

    def _extract_section_from_text(self, seccion: str, text: str) -> Dict[str, Any]:
        return self._llm_json(SECCIONES_PROMPT, seccion=seccion, text=(text or "")[:16000]) or {}

    def _extract_section_with_date_priority(
        self,
        seccion: str,
        versioning_results: Dict[str, List[Dict[str, Any]]],
        ocr_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extrae información de una sección siguiendo prioridad de fechas.
        Empieza por el documento más reciente y va hacia atrás solo para campos faltantes.
        """
        # Obtener documentos de escritura ordenados por fecha (más reciente primero)
        escritura_docs = versioning_results.get("escritura", [])
        if not escritura_docs:
            print(f"[REPORT] No hay documentos de escritura para extraer {seccion}")
            return {}
        
        # Ordenar por fecha (más reciente primero)
        escritura_docs_sorted = sorted(
            escritura_docs, 
            key=lambda x: self._parse_fecha(x.get("fecha", "")), 
            reverse=True
        )
        
        print(f"[REPORT] Extrayendo {seccion} con prioridad de fechas:")
        for i, doc in enumerate(escritura_docs_sorted):
            print(f"[REPORT]   {i+1}. {doc['filename']} ({doc['fecha']})")
        
        # Extraer texto del documento más reciente
        latest_doc = escritura_docs_sorted[0]
        latest_text = _ocr_text_by_filename(ocr_results, latest_doc["filename"])
        
        if not latest_text:
            print(f"[REPORT] No se encontró texto para {latest_doc['filename']}")
            return {}
        
        # Extraer información del documento más reciente
        print(f"[REPORT] Extrayendo {seccion} desde {latest_doc['filename']} ({latest_doc['fecha']})")
        result = self._extract_section_from_text(seccion, latest_text)
        
        # Verificar campos vacíos y buscar en documentos anteriores
        empty_fields = self._get_empty_fields(result)
        
        if empty_fields:
            print(f"[REPORT] Campos faltantes en {latest_doc['filename']}: {empty_fields}")
            
            # Buscar en documentos anteriores solo para campos faltantes
            for doc in escritura_docs_sorted[1:]:
                if not empty_fields:  # Si ya no hay campos vacíos, parar
                    break
                    
                doc_text = _ocr_text_by_filename(ocr_results, doc["filename"])
                if not doc_text:
                    continue
                
                print(f"[REPORT] Buscando campos faltantes en {doc['filename']} ({doc['fecha']})")
                doc_result = self._extract_section_from_text(seccion, doc_text)
                
                # Llenar solo campos que estén vacíos
                for field in empty_fields[:]:  # Copiar lista para iterar
                    if doc_result.get(field) and doc_result[field].strip():
                        result[field] = doc_result[field]
                        empty_fields.remove(field)
                        print(f"[REPORT]   ✓ Campo '{field}' encontrado en {doc['filename']}")
                
                if not empty_fields:
                    print(f"[REPORT] Todos los campos completados")
                    break
        
        return result

    def _get_empty_fields(self, data: Dict[str, Any]) -> List[str]:
        """Retorna lista de campos que están vacíos o no existen."""
        empty_fields = []
        for key, value in data.items():
            if not value or (isinstance(value, str) and not value.strip()):
                empty_fields.append(key)
        return empty_fields

    def _parse_fecha(self, fecha_str: str) -> datetime:
        """Convierte fecha en formato dd-mm-aaaa a datetime para ordenamiento."""
        try:
            return datetime.strptime(fecha_str, "%d-%m-%Y")
        except:
            return datetime.min  # Fecha mínima si no se puede parsear

    def _compose_text_for_sections(
        self,
        versioning_results: Dict[str, List[Dict[str, Any]]],
        ocr_results: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Elige textos recientes para extraer secciones.
        - constitucion: última versión de 'constitucion'
        - capital_social: usa constitucion reciente, si no, publicacion/inscripcion reciente
        - administracion: idem capital_social
        - legalizacion: mezcla constitucion + publicacion + inscripcion (últimas)
        """
        from src.config import should_extract_from_ocr
        
        texts = {"constitucion":"", "capital_social":"", "administracion":"", "legalizacion":""}

        latest_const = _select_latest_by_category(versioning_results, "constitucion")
        latest_pub   = _select_latest_by_category(versioning_results, "publicacion")
        latest_insc  = _select_latest_by_category(versioning_results, "inscripcion")
        latest_mod   = _select_latest_by_category(versioning_results, "modificacion")
        latest_ext   = _select_latest_by_category(versioning_results, "extracto")

        if should_extract_from_ocr():
            # Modo OCR directo: usar más documentos y combinar información
            print("[REPORT] Modo OCR directo: combinando múltiples documentos para mejor extracción")
            
            # Constitución: combinar constitución, modificación y transformación
            const_parts = []
            if latest_const:
                const_parts.append(_ocr_text_by_filename(ocr_results, latest_const["filename"]))
            if latest_mod:
                const_parts.append(_ocr_text_by_filename(ocr_results, latest_mod["filename"]))
            texts["constitucion"] = "\n\n---\n\n".join([p for p in const_parts if p])
            
            # Capital social y administración: usar constitución + extracto
            capital_parts = []
            if texts["constitucion"]:
                capital_parts.append(texts["constitucion"])
            if latest_ext:
                capital_parts.append(_ocr_text_by_filename(ocr_results, latest_ext["filename"]))
            texts["capital_social"] = "\n\n---\n\n".join([p for p in capital_parts if p])
            texts["administracion"] = texts["capital_social"]

            # Legalización: combinar todos los documentos relevantes
            legal_parts = []
            if latest_const:
                legal_parts.append(_ocr_text_by_filename(ocr_results, latest_const["filename"]))
            if latest_mod:
                legal_parts.append(_ocr_text_by_filename(ocr_results, latest_mod["filename"]))
            if latest_ext:
                legal_parts.append(_ocr_text_by_filename(ocr_results, latest_ext["filename"]))
            if latest_pub:
                legal_parts.append(_ocr_text_by_filename(ocr_results, latest_pub["filename"]))
            if latest_insc:
                legal_parts.append(_ocr_text_by_filename(ocr_results, latest_insc["filename"]))
            texts["legalizacion"] = "\n\n---\n\n".join([p for p in legal_parts if p])

        else:
            # Modo original: usar lógica más conservadora
            if latest_const:
                texts["constitucion"] = _ocr_text_by_filename(ocr_results, latest_const["filename"])
                # capital/adm prefieren constitución reciente
                texts["capital_social"] = texts["constitucion"]
                texts["administracion"] = texts["constitucion"]

            # Legalización: conviene aportar señales de notaría/DO/inscripción
            parts = []
            if latest_const:
                parts.append(_ocr_text_by_filename(ocr_results, latest_const["filename"]))
            if latest_pub:
                parts.append(_ocr_text_by_filename(ocr_results, latest_pub["filename"]))
            if latest_insc:
                parts.append(_ocr_text_by_filename(ocr_results, latest_insc["filename"]))
            texts["legalizacion"] = "\n\n---\n\n".join([p for p in parts if p])

        # Si no hubo constitución (caso raro), cae al más reciente disponible
        if not texts["capital_social"]:
            texts["capital_social"] = texts["legalizacion"]
        if not texts["administracion"]:
            texts["administracion"] = texts["legalizacion"]

        return texts

    def _build_poderes(
        self,
        versioning_results: Dict[str, List[Dict[str, Any]]],
        ocr_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Usa textos más recientes (constitución/modificación; si no hay, insc/publicación)
        y cruza contra la tabla de facultades. Devuelve:
        {"poderes_y_personerias":[{"grupo","codigo","nombre","descripcion"}...]}
        """
        tabla = self._fetch_facultades()
        if not tabla:
            return {"poderes_y_personerias": []}

        # Pool de textos recientes (2 por categoría)
        pool = _latest_texts_pool(versioning_results, ocr_results, top_per_cat=2)
        # Preferimos constitucion
        texto_base = ""
        for cat, fn, tx in pool:
            if cat == "constitucion" and tx:
                texto_base += f"\n\n[constitucion:{fn}]\n{tx}"
        # si quedó vacío, intenta con inscripcion y publicacion
        if not texto_base:
            for cat, fn, tx in pool:
                if cat in ("inscripcion","publicacion") and tx:
                    texto_base += f"\n\n[{cat}:{fn}]\n{tx}"

        # Armar tabla para el prompt (texto simple)
        tabla_txt = "\n".join([
            f"- {t['grupo']} | {t['codigo']} | {t['nombre']} | {t['descripcion']} | {t['palabras_claves']}"
            for t in tabla
        ])[:16000]

        data = self._llm_json(PODERES_PROMPT, tabla=tabla_txt, text=texto_base[:16000]) or {}
        # Sanitizar salida mínima
        items = data.get("poderes_y_personerias") or []
        out = []
        for it in items:
            grp = (it.get("grupo") or "").strip()
            cod = (it.get("codigo") or "").strip()
            nom = (it.get("nombre") or "").strip()
            des = (it.get("descripcion") or "").strip()
            if grp and cod and nom:
                out.append({"grupo":grp, "codigo":cod, "nombre":nom, "descripcion":des})
        return {"poderes_y_personerias": out}

    def _build_restricciones(
        self,
        versioning_results: Dict[str, List[Dict[str, Any]]],
        ocr_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Busca restricciones explícitas en los textos de las últimas versiones.
        Mapea a códigos si es posible (el LLM lo intentará; si no, deja []).
        """
        pool = _latest_texts_pool(versioning_results, ocr_results, top_per_cat=2)
        texto = ""
        for _, fn, tx in pool:
            if tx:
                texto += f"\n\n[{fn}]\n{tx}"
        data = self._llm_json(RESTRICCIONES_PROMPT, text=texto[:16000]) or {}
        # Sanitizar mínima
        items = data.get("restricciones") or []
        clean = []
        for it in items:
            desc = (it.get("descripcion") or "").strip()
            facs = it.get("facultades_afectadas") or []
            if desc:
                clean.append({"descripcion": desc, "facultades_afectadas": [str(x) for x in facs]})
        return {"restricciones": clean}

    # -------------- API principal --------------
    def generate_complete_report(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        all_results:
          {
            "versioning_results": {...},
            "comparison_results": {...},          # (opcional para enriquecer)
            "entity_results": {...},
            "ocr_results": [...],
            "date_results": [...],
            "classification_results": [...]
          }
        """
        from src.config import should_extract_from_ocr
        
        entity = all_results.get("entity_results", {}) or {}
        versioning = all_results.get("versioning_results", {}) or {}
        ocr = all_results.get("ocr_results", []) or []

        report: Dict[str, Any] = {}

        # Verificar configuración de extracción
        extract_from_ocr = should_extract_from_ocr()
        print(f"[REPORT] Modo de extracción: {'OCR directo' if extract_from_ocr else 'Sin vectorización'}")

        # 1) Encabezado con prioridad de fechas
        print("[REPORT] Extrayendo encabezado con prioridad de fechas...")
        encabezado = self._extract_encabezado_with_date_priority(versioning, ocr)
        report.update(encabezado)

        # Ambos modos extraen desde OCR directamente (no usan vectores)
        if extract_from_ocr:
            print("[REPORT] Extrayendo información directamente desde OCR (modo optimizado)...")
        else:
            print("[REPORT] Extrayendo información directamente desde OCR (modo sin vectorización)...")
        
        # 2) Secciones sustantivas con prioridad de fechas
        print("[REPORT] Iniciando extracción con prioridad de fechas...")
        
        seccion_const = self._extract_section_with_date_priority("constitucion", versioning, ocr)
        seccion_capital = self._extract_section_with_date_priority("capital_social", versioning, ocr)
        seccion_admin = self._extract_section_with_date_priority("administracion", versioning, ocr)
        
        # Para legalización, usar el método original ya que necesita información de múltiples categorías
        texts = self._compose_text_for_sections(versioning, ocr)
        seccion_legal = self._extract_section_from_text("legalizacion", texts["legalizacion"])

        report["seccion_constitucion"] = seccion_const
        report["seccion_capital_social"] = seccion_capital
        report["seccion_administracion"] = seccion_admin
        report["legalizacion"] = seccion_legal

        # 4) Poderes y Personerías
        poderes = self._build_poderes(versioning, ocr)
        report["poderes_y_personerias"] = poderes.get("poderes_y_personerias", [])

        # 5) Restricciones
        restricciones = self._build_restricciones(versioning, ocr)
        report["restricciones"] = restricciones.get("restricciones", [])

        return report
