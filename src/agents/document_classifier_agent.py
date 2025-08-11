# src/agents/document_classifier_agent.py
from typing import List, Dict, Any
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

CATS = ["constitucion", "inscripcion", "publicacion", "cedula", "otros"]

# Palabras clave de categoría base
KW = {
    "constitucion": [
        "escritura pública", "escritura publica", "instrumento privado protocolizado",
        "acta protocolizada", "estatutos", "acta de directorio",
        "designación y poderes", "designacion y poderes", "poderes otorgados",
        "carta de los socios", "antecedentes que acrediten", "objeto social",
        "domicilio de la agencia", "extracto de la constitución", "extracto de modific",
        "extracto", "constitución", "constituida", "constituye", "constituyen",
        "sociedad anónima", "sociedad limitada", "spa", "eirl", "sociedad por acciones",
        "capital suscrito", "capital pagado", "razón social", "razon social",
        "objeto social", "domicilio", "duración", "administración", "administradores",
        "representante legal", "representantes legales", "socios", "acciones",
        "participaciones", "estatutos sociales", "acta de constitución"
    ],
    "inscripcion": [
        "inscripción", "inscripcion", "registro de comercio", "conservador",
        "anotaciones marginales", "certificado de vigencia", "registro de propiedad",
        "interdicciones", "fojas"
    ],
    "publicacion": [
        "publicación", "publicacion", "diario oficial"
    ],
    "cedula": [
        "cédula de identidad", "cedula de identidad"
    ],
}

# Palabras clave que indican que el documento es una modificación
KW_MODIFICACION = [
    "modificación", "modificacion", "reforma", "aumento de capital", "disminución de capital",
    "ingreso de socio", "retiro de socio", "cambio de nombre", "cambio de razón social",
    "cambio de domicilio", "cambio de objeto social", "agregar restriccion", "agregar restricciones",
    "modificación de estatutos", "modificacion de estatutos"
]

LLM_PROMPT = ChatPromptTemplate.from_template("""
Eres un experto en documentos legales chilenos. Clasifica el documento en UNA de estas categorías según su contenido (NO el nombre del archivo):

- constitucion: Documentos que establecen o modifican la estructura legal de una empresa:
  * Escrituras públicas de constitución
  * Extractos de constitución o modificación de estatutos
  * Actas de directorio o socios
  * Estatutos sociales
  * Documentos que mencionen capital, socios, administración, objeto social
  * Cualquier documento que defina la estructura legal de la empresa

- inscripcion: Documentos de registro oficial:
  * Inscripciones en Registro de Comercio
  * Certificados de vigencia
  * Anotaciones marginales
  * Documentos del Conservador de Bienes Raíces

- publicacion: Publicaciones oficiales:
  * Publicaciones en Diario Oficial (aunque contengan extractos de constitución)
  * Avisos legales obligatorios
  * Cualquier documento que sea claramente una publicación oficial

- cedula: Documentos de identificación:
  * Cédulas de identidad
  * Documentos de identificación personal

- otros: Cualquier otro tipo de documento

IMPORTANTE: 
- Si el documento es claramente una publicación del Diario Oficial, clasifícalo como "publicacion", aunque contenga extractos de constitución
- Los extractos de constitución SOLOS van en "constitucion"
- Las publicaciones del Diario Oficial que contienen extractos van en "publicacion"

Responde EXCLUSIVAMENTE con una palabra: constitucion | inscripcion | publicacion | cedula | otros

TEXTO:
----------------
{text}
----------------
""".strip())

class DocumentClassifierAgent:
    def __init__(self, get_llm_callable=None):
        if get_llm_callable:
            self.llm = get_llm_callable()
        else:
            from src.config import get_llm
            self.llm = get_llm()
        self.prompt = LLM_PROMPT
        self.parser = StrOutputParser()

    def _rule_based(self, text: str) -> str:
        t = (text or "").lower()
        if any(k in t for k in KW["cedula"]):
            return "cedula"
        if any(k in t for k in KW["publicacion"]):
            return "publicacion"
        if any(k in t for k in KW["inscripcion"]):
            return "inscripcion"
        if any(k in t for k in KW["constitucion"]):
            return "constitucion"
        return "otros"

    def _detect_modificacion(self, text: str) -> bool:
        """Detecta si el documento indica modificación sin cambiar la categoría."""
        t = (text or "").lower()
        return any(k in t for k in KW_MODIFICACION)

    def _llm_fallback(self, text: str) -> str:
        msgs = self.prompt.format_messages(text=(text or "")[:16000])
        raw = self.llm.invoke(msgs)
        if hasattr(raw, "content"):
            raw = raw.content
        cand = (raw or "").strip().lower()
        return cand if cand in CATS else "otros"

    def classify_documents(self, documents_with_dates: List[Dict[str, Any]], tipo_empresa: str) -> List[Dict[str, Any]]:
        """
        Entrada: [{filename, text, fecha}, ...]
        Salida:  [{filename, clasificacion, fecha, is_modificacion}, ...]
        """
        out = []
        for d in documents_with_dates:
            text = d.get("text", "")
            
            # Usar directamente el LLM para clasificar
            clasif = self._llm_fallback(text)
            is_modif = self._detect_modificacion(text)

            out.append({
                "filename": d.get("filename"),
                "clasificacion": clasif,
                "fecha": d.get("fecha", ""),
                "is_modificacion": is_modif
            })
        return out
