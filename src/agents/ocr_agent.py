# src/agents/ocr_agent.py

import os
import hashlib
import time
import pickle
from typing import List, Dict, Optional

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential

ALLOWED_EXT = None  # None = intenta con todo; Document Intelligence soporta PDF/PNG/JPG/TIFF

def _sha1_bytes(b: bytes) -> str:
    h = hashlib.sha1()
    h.update(b)
    return h.hexdigest()

def _clean_soft(text: str) -> str:
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines()]
    non_empty = []
    prev_blank = False
    for ln in lines:
        if ln == "":
            if not prev_blank:
                non_empty.append("")
            prev_blank = True
        else:
            non_empty.append(ln)
            prev_blank = False
    cleaned = "\n".join(non_empty)
    cleaned = "\n".join(" ".join(line.split()) for line in cleaned.split("\n"))
    return cleaned.strip()

class OCRAgent:
    def __init__(self, model: str = "prebuilt-read", max_retries: int = 4, cache_dir: str = "ocr_cache"):
        # Obtener credenciales desde variables de entorno
        endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY")
        
        if not endpoint or not key:
            raise ValueError("Azure Document Intelligence endpoint/key no configurados. Verifica las variables de entorno AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT y AZURE_DOCUMENT_INTELLIGENCE_API_KEY.")
        
        self.client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
        self.model = model
        self.max_retries = max_retries
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, sha1_hash: str) -> str:
        return os.path.join(self.cache_dir, f"{sha1_hash}.pkl")

    def _load_from_cache(self, sha1_hash: str) -> Optional[Dict]:
        cache_path = self._get_cache_path(sha1_hash)
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    cached_result = pickle.load(f)
                print(f"[OCR][CACHE] Cargado desde caché: {cached_result.get('filename', '')}")
                return cached_result
        except Exception as e:
            print(f"[OCR][CACHE-ERROR] {e}")
        return None

    def _save_to_cache(self, sha1_hash: str, result: Dict):
        cache_path = self._get_cache_path(sha1_hash)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            print(f"[OCR][CACHE] Guardado en caché: {result.get('filename', '')}")
        except Exception as e:
            print(f"[OCR][CACHE-ERROR] {e}")

    def _analyze_file(self, file_path: str) -> Optional[Dict]:
        try:
            with open(file_path, "rb") as f:
                content = f.read()
        except Exception as e:
            print(f"[OCR][SKIP] No se pudo leer {file_path}: {e}")
            return None

        filename = os.path.basename(file_path)
        file_ext = os.path.splitext(filename)[1].lower()
        filesize = len(content)
        sha1 = _sha1_bytes(content)

        cached_result = self._load_from_cache(sha1)
        if cached_result:
            return cached_result

        if ALLOWED_EXT and file_ext not in ALLOWED_EXT:
            print(f"[OCR][SKIP] Extensión no soportada para {filename}")
            return None
            
        # Verificar si el archivo es muy grande y podría tener problemas con límites de páginas
        if filesize > 2000000:  # Más de 2MB
            print(f"[OCR] Archivo muy grande detectado: {filename} ({filesize} bytes)")
            print(f"[OCR] Considera dividir el documento en partes más pequeñas si hay problemas de límites")

        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                # No limitamos páginas → lee todo el documento
                req = AnalyzeDocumentRequest(bytes_source=content)
                poller = self.client.begin_analyze_document(self.model, body=req)
                print(f"[OCR] Procesando {filename} ({filesize} bytes)")
                result = poller.result()
                
                # Verificar si hay límites de páginas
                if hasattr(result, "pages") and result.pages:
                    detected_pages = len(result.pages)
                    print(f"[OCR] Páginas detectadas: {detected_pages}")
                    
                    # Si el archivo es grande pero solo detectamos pocas páginas, puede ser un límite del plan
                    if filesize > 500000 and detected_pages <= 2:  # Más de 500KB y solo 2 páginas
                        print(f"[OCR] ADVERTENCIA: Archivo grande ({filesize} bytes) pero solo {detected_pages} páginas detectadas")
                        print(f"[OCR] Esto puede indicar un límite del plan de Azure Document Intelligence")
                        print(f"[OCR] Considera actualizar el plan para procesar más páginas por documento")

                full_content = getattr(result, "content", "") or ""
                pages = []
                if hasattr(result, "pages") and result.pages:
                    print(f"[OCR] Documento tiene {len(result.pages)} páginas detectadas")
                    for p in result.pages:
                        page_text_parts = []
                        spans = getattr(p, "spans", []) or []
                        if spans and full_content:
                            for sp in spans:
                                off = getattr(sp, "offset", 0) or 0
                                ln = getattr(sp, "length", 0) or 0
                                page_text_parts.append(full_content[off: off + ln])
                            page_text = "".join(page_text_parts).strip()
                        else:
                            line_items = getattr(p, "lines", []) or []
                            page_text = "\n".join(
                                li.content for li in line_items if getattr(li, "content", None)
                            ).strip()
                        pages.append({
                            "page": getattr(p, "page_number", len(pages)+1),
                            "text": page_text
                        })
                else:
                    print(f"[OCR] No se detectaron páginas en el documento")

                if any(pg["text"] for pg in pages):
                    text_raw = "\n\n".join(pg["text"] for pg in pages if pg["text"])
                else:
                    text_raw = full_content

                text_clean = _clean_soft(text_raw)

                language = None
                try:
                    langs = getattr(result, "languages", []) or []
                    if langs:
                        language = sorted(langs, key=lambda l: getattr(l, "confidence", 0), reverse=True)[0].locale
                except Exception:
                    language = None

                result_dict = {
                    "filename": filename,
                    "path": file_path,
                    "file_ext": file_ext,
                    "filesize": filesize,
                    "sha1": sha1,
                    "num_pages": len(result.pages) if hasattr(result, "pages") else None,
                    "language": language,
                    "text": text_clean,
                    "text_raw": text_raw,
                    "pages": pages
                }
                
                print(f"[OCR] Completado {filename}: {len(result.pages) if hasattr(result, 'pages') else 0} páginas extraídas")

                self._save_to_cache(sha1, result_dict)
                return result_dict

            except Exception as e:
                msg = str(e).lower()
                if "429" in msg or "too many requests" in msg or "503" in msg or "temporarily" in msg:
                    print(f"[OCR][RETRY {attempt+1}] {filename} Esperando {delay:.1f}s…")
                    time.sleep(delay)
                    delay = min(delay * 2, 10.0)
                    continue
                print(f"[OCR][ERROR] {filename}: {e}")
                return None

    def analyze_folder(self, folder_path: str) -> List[Dict]:
        if not os.path.isdir(folder_path):
            raise ValueError(f"Directorio no existe: {folder_path}")

        results: List[Dict] = []
        total_pages = 0
        large_files_with_few_pages = 0
        
        for root, _, files in os.walk(folder_path):
            for name in files:
                path = os.path.join(root, name)
                item = self._analyze_file(path)
                if item:
                    results.append(item)
                    total_pages += item.get("num_pages", 0)
                    
                    # Contar archivos grandes con pocas páginas
                    if item.get("filesize", 0) > 500000 and item.get("num_pages", 0) <= 2:
                        large_files_with_few_pages += 1
        
        print(f"\n[OCR] RESUMEN:")
        print(f"[OCR] Total documentos procesados: {len(results)}")
        print(f"[OCR] Total páginas extraídas: {total_pages}")
        print(f"[OCR] Archivos grandes con pocas páginas: {large_files_with_few_pages}")
        
        if large_files_with_few_pages > 0:
            print(f"\n[OCR] RECOMENDACIONES:")
            print(f"[OCR] - Tu plan actual de Azure Document Intelligence parece tener límites de páginas")
            print(f"[OCR] - Considera actualizar a un plan que permita más páginas por documento")
            print(f"[OCR] - Planes gratuitos suelen limitar a 2 páginas por documento")
            print(f"[OCR] - Planes de pago permiten hasta 2000 páginas por documento")
            print(f"[OCR] - Verifica tu plan en el portal de Azure")
        
        return results
