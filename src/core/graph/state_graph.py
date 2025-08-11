"""
StateGraph para el procesamiento de documentos legales usando LangGraph.
"""

import os
import json
from typing import TypedDict, Any
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from src.agents.ocr_agent import OCRAgent

from src.agents.document_classifier_agent import DocumentClassifierAgent
from src.agents.legalization_agent import LegalizationAgent
from src.agents.vectorization_agent import VectorizationAgent
from src.agents.versioning_agent import VersioningAgent
from src.agents.version_comparer_agent import VersionComparerAgent
from src.agents.report_generator_agent import ReportGeneratorAgent

# Cargar variables de entorno
load_dotenv()

# --- Definición del State ---
class ProcessingState(TypedDict):
    """Estado del procesamiento."""
    carpeta: str
    folder_path: str
    ocr_results: list
    entity_results: dict
    classification_results: list
    vectorization_results: dict
    versioning_results: dict
    comparison_results: dict
    report_results: dict
    error: str

def create_initial_state(carpeta: str) -> ProcessingState:
    """Crea el estado inicial."""
    folder_path = os.path.join("data", carpeta)
    return {
        "carpeta": carpeta,
        "folder_path": folder_path,
        "ocr_results": [],
        "entity_results": {},
        "classification_results": [],
        "vectorization_results": {},
        "versioning_results": {},
        "comparison_results": {},
        "legalization_results": {},
        "report_results": {},
        "error": ""
    }

# --- NODOS DEL GRAFO ---
def paso_1_ocr(state: ProcessingState) -> ProcessingState:
    carpeta = state["carpeta"]
    folder_path = state["folder_path"]

    print("=" * 50)
    print(f"PASO 1: OCR - Procesando carpeta {carpeta}")
    print("=" * 50)

    try:
        os.makedirs("results", exist_ok=True)  # <-- crea carpeta si no exist

        ocr_agent = OCRAgent()
        results = ocr_agent.analyze_folder(folder_path)

        print(f"OCR completado: {len(results)} documentos")

        results_file = os.path.join("results", f"{carpeta}_ocr_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Resultados guardados en: {results_file}")

        return {**state, "ocr_results": results}

    except Exception as e:
        error_msg = f"Error en PASO 1: {e}"
        print(error_msg)
        return {**state, "error": error_msg}


def paso_2_versioning(state: ProcessingState) -> ProcessingState:
    """PASO 2: Versionado de documentos."""
    carpeta = state["carpeta"]
    ocr_results = state["ocr_results"]

    if not ocr_results:
        error_msg = "No hay resultados de OCR para versionar documentos"
        print(error_msg)
        return {**state, "error": error_msg}

    print("=" * 50)
    print(f"PASO 2: Versionado de documentos - {carpeta}")
    print("=" * 50)

    try:
        versioning_agent = VersioningAgent()

        versioning_results = versioning_agent.assign_versions_from_ocr(ocr_results)

        # Mostrar en consola
        versioning_agent.mostrar_versionado(versioning_results)

        # Guardar resultados
        os.makedirs("results", exist_ok=True)
        results_file = os.path.join("results", f"{carpeta}_versioning_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(versioning_results, f, ensure_ascii=False, indent=2)

        print("Versionado completado")
        total_docs = sum(len(v) for v in versioning_results.values())
        print(f"Total documentos versionados: {total_docs}")
        print(f"Resultados guardados en: {results_file}")

        return {**state, "versioning_results": versioning_results}

    except Exception as e:
        error_msg = f"Error en PASO 2: {e}"
        print(error_msg)
        return {**state, "error": error_msg}




def paso_4_escritura_comparison(state: ProcessingState) -> ProcessingState:
    """PASO 4: Comparación de versiones de documentos de escritura."""
    carpeta = state["carpeta"]
    versioning_results = state["versioning_results"]
    ocr_results = state["ocr_results"]

    if not versioning_results:
        error_msg = "No hay resultados de versionado para comparar"
        print(error_msg)
        return {**state, "error": error_msg}

    print("=" * 50)
    print(f"PASO 4: Comparación de versiones de escrituras - {carpeta}")
    print("=" * 50)

    try:
        # Filtrar solo documentos de categoría "escritura"
        escrituras = versioning_results.get("escritura", [])
        
        if not escrituras:
            print("No se encontraron documentos de escritura para comparar")
            return {**state, "comparison_results": {}}

        # Ordenar por versión
        escrituras_ordenadas = sorted(escrituras, key=lambda x: x.get("version", 0))
        
        print(f"Documentos de escritura encontrados: {len(escrituras_ordenadas)}")
        for esc in escrituras_ordenadas:
            print(f"  - {esc['filename']} (fecha {esc['fecha']}) --> versión {esc['version']}")

        # Crear índice de texto OCR por filename
        ocr_index = {doc["filename"]: doc.get("text", "") for doc in ocr_results}

        # Comparar versiones consecutivas
        comparison_results = []
        
        for i in range(len(escrituras_ordenadas) - 1):
            version_actual = escrituras_ordenadas[i]
            version_siguiente = escrituras_ordenadas[i + 1]
            
            print(f"\nComparando:")
            print(f"  De: {version_actual['filename']} (fecha {version_actual['fecha']}) --> versión {version_actual['version']}")
            print(f"  A: {version_siguiente['filename']} (fecha {version_siguiente['fecha']}) --> versión {version_siguiente['version']}")
            
            # Obtener textos
            texto_actual = ocr_index.get(version_actual["filename"], "")
            texto_siguiente = ocr_index.get(version_siguiente["filename"], "")
            
            if not texto_actual or not texto_siguiente:
                print("  ⚠️ No se encontró texto OCR para una de las versiones")
                continue
            
            # Usar VersionComparerAgent para comparar
            comparison_agent = VersionComparerAgent()
            
            # Crear estructura temporal para el agente
            temp_versioning = {
                "escritura": [version_actual, version_siguiente]
            }
            
            try:
                # Comparar usando el agente existente
                comparison = comparison_agent.comparar_versiones(
                    versioning_results=temp_versioning,
                    ocr_results=ocr_results
                )
                
                # Extraer resultados de la comparación
                if "escritura" in comparison and comparison["escritura"]:
                    # Simplificar la estructura del JSON
                    comparacion_simple = comparison["escritura"][0] if comparison["escritura"] else {}
                    
                    comparison_results.append({
                        "categoria": "escritura",
                        "de": {
                            "filename": version_actual["filename"],
                            "fecha": version_actual["fecha"],
                            "version": version_actual["version"]
                        },
                        "a": {
                            "filename": version_siguiente["filename"],
                            "fecha": version_siguiente["fecha"],
                            "version": version_siguiente["version"]
                        },
                        "cambios": comparacion_simple.get("diferencias", {}).get("cambios", []),
                        "resumen": comparacion_simple.get("diferencias", {}).get("resumen", "")
                    })
                    
                    print("  Comparación completada")
                else:
                    print("  No se detectaron diferencias significativas")
                    
            except Exception as e:
                print(f"  Error en comparación: {e}")
                continue

        # Guardar resultados
        os.makedirs("results", exist_ok=True)
        results_file = os.path.join("results", f"{carpeta}_escritura_comparison_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=2)

        print(f"\nComparación de escrituras completada")
        print(f"Total comparaciones realizadas: {len(comparison_results)}")
        print(f"Resultados guardados en: {results_file}")

        return {**state, "comparison_results": {"escritura": comparison_results}}

    except Exception as e:
        error_msg = f"Error en PASO 4: {e}"
        print(error_msg)
        return {**state, "error": error_msg}


def paso_5_legalizacion(state: ProcessingState) -> ProcessingState:
    carpeta = state["carpeta"]
    ocr_results = state["ocr_results"]
    versioning_results = state["versioning_results"]

    if not ocr_results or not versioning_results:
        print("Faltan OCR o versionado para legalización")
        return state

    print("=" * 50)
    print(f"PASO 5: Legalización (poderes) - {carpeta}")
    print("=" * 50)

    try:
        import os, json
        from src.agents.legalization_agent import LegalizationAgent

        # --- SWITCH: prende/apaga búsqueda en vectores ---
        USE_VECTORES = True  # pon False si quieres que NO use el vectorstore

        # 1) crea el agente (por ahora sin retriever)
        agent = LegalizationAgent(
            get_llm_callable=None,      # usa src.config.get_llm()
            retriever_callable=None,    # se setea abajo si USE_VECTORES=True
            max_chars_doc=16000,
            min_similarity=0.70
        )

        # 2) si quieres usar vectores, asigna el retriever del propio agente
        if USE_VECTORES:
            # usa tu método definido dentro del agente: mi_retriever_vectorstore(...)
            def retriever(query: str, k: int):
                return agent.mi_retriever_vectorstore(
                    collection=f"legal_{carpeta}", query=query, k=k
                )
            agent.retriever = retriever  # <- lo enchufas acá

        # 3) carga el catálogo desde tu BD (método del agente)
        code_catalog = agent.cargar_catalogo_desde_bd()

        # 4) corre la extracción sobre la ÚLTIMA escritura
        results = agent.extract_from_documents(
            ocr_docs=ocr_results,
            versioning_results=versioning_results,
            code_catalog=code_catalog,
            only_latest_escritura=True
        )

        # 5) persistir resultados
        os.makedirs("results", exist_ok=True)
        out = os.path.join("results", f"{carpeta}_legalizacion_results.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Legalización: {len(results.get('poderes', []))} códigos evaluados. Guardado en {out}")
        return {**state, "legalization_results": results}

    except Exception as e:
        msg = f"Error en PASO 5 (legalización): {e}"
        print(msg)
        return {**state, "error": msg}




def paso_3_vectorization(state: ProcessingState) -> ProcessingState:
    """PASO 3: Vectorización de documentos en pgvector (con metadatos y versión)."""
    import os, json
    from src.config import should_extract_from_ocr
    
    carpeta = state["carpeta"]

    ocr_results = state["ocr_results"]                 # [{filename, text, ...}]
    versioning_results = state["versioning_results"]   # {cat: [{filename, fecha, version}, ...]}

    if not ocr_results or not versioning_results:
        error_msg = "Faltan resultados (OCR o versionado) para vectorizar"
        print(error_msg)
        return {**state, "error": error_msg}

    # Verificar configuración de extracción
    extract_from_ocr = should_extract_from_ocr()
    
    print("=" * 50)
    print(f"PASO 3: Vectorización - {carpeta}")
    print(f"Modo de extracción: {'OCR directo' if extract_from_ocr else 'Sin vectorización'}")
    print("=" * 50)

    # Si está en modo OCR directo (true), omitir vectorización
    if extract_from_ocr:
        print("[VECTORIZATION] Modo OCR directo detectado - omitiendo vectorización")
        print("[VECTORIZATION] Los documentos se procesarán directamente desde OCR")
        
        # Crear resultado vacío para mantener compatibilidad
        vec_results = {
            "collection": f"legal_{carpeta}",
            "documentos_procesados": 0,
            "total_chunks": 0,
            "modo": "OCR_DIRECTO",
            "mensaje": "Vectorización omitida - usando extracción directa desde OCR"
        }
        
        # Guardar resumen
        os.makedirs("results", exist_ok=True)
        results_file = os.path.join("results", f"{carpeta}_vectorization_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(vec_results, f, ensure_ascii=False, indent=2)

        print("Vectorización omitida (modo OCR directo)")
        print(f"Resultados guardados en: {results_file}")

        return {**state, "vectorization_results": vec_results}

    # Si está en modo false, también omitir vectorización (no vectorizar)
    print("[VECTORIZATION] Modo sin vectorización detectado - omitiendo vectorización")
    print("[VECTORIZATION] Los documentos se procesarán directamente desde OCR")
    
    # Crear resultado vacío para mantener compatibilidad
    vec_results = {
        "collection": f"legal_{carpeta}",
        "documentos_procesados": 0,
        "total_chunks": 0,
        "modo": "SIN_VECTORIZACION",
        "mensaje": "Vectorización omitida - usando extracción directa desde OCR"
    }
    
    # Guardar resumen
    os.makedirs("results", exist_ok=True)
    results_file = os.path.join("results", f"{carpeta}_vectorization_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(vec_results, f, ensure_ascii=False, indent=2)

    print("Vectorización omitida (modo sin vectorización)")
    print(f"Resultados guardados en: {results_file}")

    return {**state, "vectorization_results": vec_results}


def paso_7_version_comparison(state: ProcessingState) -> ProcessingState:
    """PASO 7: Comparación de versiones de documentos."""
    carpeta = state["carpeta"]
    versioning_results = state["versioning_results"]
    ocr_results = state["ocr_results"]

    if not versioning_results:
        error_msg = "No hay resultados de versionado para comparar"
        print(error_msg)
        return {**state, "error": error_msg}

    print("=" * 50)
    print(f"PASO 7: Comparación de versiones - {carpeta}")
    print("=" * 50)

    try:
        comparison_agent = VersionComparerAgent()

        comparison_results = comparison_agent.comparar_versiones(
            versioning_results=versioning_results,
            ocr_results=ocr_results
        )

        # Mostrar comparaciones de forma legible
        comparison_agent.mostrar_comparaciones(comparison_results)

        # Guardar
        os.makedirs("results", exist_ok=True)
        results_file = os.path.join("results", f"{carpeta}_comparison_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=2)

        print("Comparación de versiones completada")
        total_comparisons = sum(len(v) for v in comparison_results.values())
        print(f"Total comparaciones realizadas: {total_comparisons}")
        print(f"Resultados guardados en: {results_file}")

        return {**state, "comparison_results": comparison_results}

    except Exception as e:
        error_msg = f"Error en PASO 7: {e}"
        print(error_msg)
        return {**state, "error": error_msg}


def paso_8_report_generation(state: ProcessingState) -> ProcessingState:
    """PASO 8: Generación del informe legal completo."""
    import os, json
    carpeta = state["carpeta"]

    all_results = {
        "versioning_results": state.get("versioning_results", {}),
        "comparison_results": state.get("comparison_results", {}),
        "legalization_results": state.get("legalization_results", {}),
        "ocr_results": state.get("ocr_results", []),
        "vectorization_results": state.get("vectorization_results", {})
    }

    if not all_results["versioning_results"]:
        error_msg = "No hay resultados de versionado para generar informe"
        print(error_msg)
        return {**state, "error": error_msg}
    
    # Verificar que tenemos al menos algunos resultados para el informe
    has_data = any([
        all_results["versioning_results"],
        all_results["comparison_results"],
        all_results["legalization_results"],
        all_results["ocr_results"]
    ])
    
    if not has_data:
        error_msg = "No hay datos suficientes para generar el informe"
        print(error_msg)
        return {**state, "error": error_msg}

    print("=" * 50)
    print(f"PASO 8: Generación de informe legal - {carpeta}")
    print("=" * 50)

    try:
        from src.agents.report_generator_agent import ReportGeneratorAgent
        report_agent = ReportGeneratorAgent()

        report_results = report_agent.generate_complete_report(all_results)

        os.makedirs("results", exist_ok=True)
        results_file = os.path.join("results", f"{carpeta}_report_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(report_results, f, ensure_ascii=False, indent=2)

        print("Informe legal generado exitosamente")
        print(f"Secciones incluidas: {len(report_results.keys())}")
        print(f"Resultados guardados en: {results_file}")

        return {**state, "report_results": report_results}

    except Exception as e:
        error_msg = f"Error en PASO 8: {e}"
        print(error_msg)
        return {**state, "error": error_msg}


# --- CONSTRUCCIÓN DEL GRAFO ---
def build_processing_graph():
    """Construye el grafo de procesamiento."""
    workflow = StateGraph(ProcessingState)
    
    workflow.add_node("paso_1_ocr", paso_1_ocr)
    workflow.add_node("paso_2_versioning", paso_2_versioning)
    workflow.add_node("paso_3_vectorization", paso_3_vectorization)
    workflow.add_node("paso_4_escritura_comparison", paso_4_escritura_comparison)
    workflow.add_node("paso_5_legalizacion", paso_5_legalizacion)
    #workflow.add_node("paso_6_vectorization", paso_6_vectorization)
    #workflow.add_node("paso_7_version_comparison", paso_7_version_comparison)
    workflow.add_node("paso_8_report_generation", paso_8_report_generation)
    
    workflow.set_entry_point("paso_1_ocr")
    workflow.add_edge("paso_1_ocr", "paso_2_versioning")
    workflow.add_edge("paso_2_versioning", "paso_3_vectorization")
    workflow.add_edge("paso_3_vectorization", "paso_4_escritura_comparison")
    workflow.add_edge("paso_4_escritura_comparison", "paso_5_legalizacion")
    workflow.add_edge("paso_5_legalizacion", "paso_8_report_generation")
    #workflow.add_edge("paso_3_date_extraction", "paso_4_entity_extraction")
    #workflow.add_edge("paso_4_entity_extraction", "paso_5_document_classification")
    #workflow.add_edge("paso_5_document_classification", "paso_6_vectorization")
    #workflow.add_edge("paso_6_vectorization", "paso_7_version_comparison")
    #workflow.add_edge("paso_7_version_comparison", "paso_8_report_generation")
    #workflow.add_edge("paso_8_report_generation", END)
    workflow.add_edge("paso_8_report_generation", END)
    
    return workflow.compile() 