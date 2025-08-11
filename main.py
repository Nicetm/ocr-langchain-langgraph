#!/usr/bin/env python3
"""
Script principal para ejecutar Poc Legales con StateGraph
"""

import sys
import os

def main():
    """Función principal del sistema."""
    
    if len(sys.argv) < 2:
        print("Uso: python main.py <nombre_carpeta>")
        sys.exit(1)
    
    carpeta = sys.argv[1]
    folder_path = os.path.join("data", carpeta)
    
    if not os.path.exists(folder_path):
        print(f"Error: La carpeta {folder_path} no existe")
        sys.exit(1)
    
    print("POC LEGALES - PROCESAMIENTO CON STATEGRAPH")
    print("=" * 50)
    
    try:
        from src.core.orchestrator.orchestrator import DocumentOrchestrator
        
        orchestrator = DocumentOrchestrator()
        results = orchestrator.process_carpeta(carpeta)
        
        if results["success"]:
            print("Proceso completado exitosamente")
        else:
            print(f"Proceso falló: {results['error']}")
            sys.exit(1)
        
    except Exception as e:
        print(f"Error crítico: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 