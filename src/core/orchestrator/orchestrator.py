"""
Orquestador principal para el procesamiento de documentos legales.
"""

import os
from typing import Dict, Any
from ...core.graph.state_graph import build_processing_graph, create_initial_state

class DocumentOrchestrator:
    """Orquestador que coordina el procesamiento usando StateGraph."""
    
    def __init__(self):
        """Inicializa el orquestador."""
        self.processing_graph = build_processing_graph()
        print("Orquestador inicializado")
    
    def process_carpeta(self, carpeta: str) -> Dict[str, Any]:
        """
        Procesa una carpeta completa usando el StateGraph.
        
        Args:
            carpeta: Nombre de la carpeta a procesar
            
        Returns:
            Dict con los resultados del procesamiento
        """
        print(f"Iniciando procesamiento: {carpeta}")
        
        try:
            initial_state = create_initial_state(carpeta)
            final_state = self.processing_graph.invoke(initial_state)
            
            results = {
                "carpeta": carpeta,
                "success": not bool(final_state.get("error")),
                "ocr_results": final_state.get("ocr_results", []),
                "entity_results": final_state.get("entity_results", {}),
                "vectorization_results": final_state.get("vectorization_results", {}),
                "versioning_results": final_state.get("versioning_results", {}),
                "comparison_results": final_state.get("comparison_results", {}),
                "legalization_results": final_state.get("legalization_results", {}),
                "report_results": final_state.get("report_results", {}),
                "error": final_state.get("error", "")
            }
            
            if results["success"]:
                print("=" * 50)
                print("RESUMEN FINAL")
                print("=" * 50)
                print(f"Documentos procesados: {len(results['ocr_results'])}")
                versioning_results = results.get('versioning_results', {})
                total_versioned = sum(len(docs) for docs in versioning_results.values())
                print(f"Documentos versionados: {total_versioned}")
                print(f"Categorías de documentos: {list(versioning_results.keys())}")
                vectorization_results = results.get('vectorization_results', {})
                print(f"Documentos vectorizados: {vectorization_results.get('documentos_procesados', 0)}")
                print(f"Total chunks: {vectorization_results.get('total_chunks', 0)}")
                comparison_results = results.get('comparison_results', {})
                total_comparisons = sum(len(comps) for comps in comparison_results.values())
                print(f"Comparaciones de escrituras: {total_comparisons}")
                legalization_results = results.get('legalization_results', {})
                poderes = legalization_results.get('poderes', [])
                print(f"Códigos legales evaluados: {len(poderes)}")
                print(f"Códigos otorgados: {len([p for p in poderes if p.get('otorgado', False)])}")
                report_results = results.get('report_results', {})
                print(f"Informe legal generado: {'SÍ' if report_results else 'NO'}")
                print("=" * 50)
            else:
                print(f"Procesamiento fallido: {results['error']}")
            
            return results
            
        except Exception as e:
            error_msg = f"Error en orquestador: {e}"
            print(error_msg)
            return {
                "carpeta": carpeta,
                "success": False,
                "ocr_results": [],
                "company_type": "",
                "error": error_msg
            } 