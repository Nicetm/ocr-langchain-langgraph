# Poc Legales - Sistema de Procesamiento de Documentos Legales

Sistema automatizado para procesar, analizar y generar reportes de documentos legales utilizando Azure Document Intelligence, Azure OpenAI y PostgreSQL con pgvector.

## 🎯 Descripción

Este sistema procesa documentos legales (escrituras públicas, inscripciones, publicaciones) y extrae información estructurada para generar reportes legales completos. Utiliza un flujo de 8 pasos automatizados que van desde el OCR hasta la generación de reportes finales.

## 🏗️ Arquitectura

### Tecnologías Utilizadas
- **Azure Document Intelligence**: OCR y extracción de texto
- **Azure OpenAI**: Análisis de texto y extracción de información
- **PostgreSQL + pgvector**: Almacenamiento de vectores y metadatos
- **LangChain**: Framework para LLMs y procesamiento de texto
- **LangGraph**: Orquestación del flujo de procesamiento
- **Docker**: Contenedorización de la base de datos

### Estructura del Proyecto
```
Poc Legales/
├── data/                          # Documentos a procesar
│   ├── BASTIAS SPA 76.154.106-4/
│   ├── SERVICIOS Y ASESORIAS.../
│   └── SOFTKEY SPA 15.060.791-4/
├── src/
│   ├── agents/                    # Agentes especializados
│   │   ├── ocr_agent.py          # PASO 1: OCR
│   │   ├── date_extractor_agent.py # PASO 2: Extracción de fechas

│   │   ├── document_classifier_agent.py # PASO 4: Clasificación
│   │   ├── vectorization_agent.py # PASO 5: Vectorización
│   │   ├── versioning_agent.py   # PASO 5: Versionado
│   │   ├── version_comparer_agent.py # PASO 6: Comparación
│   │   └── report_generator_agent.py # PASO 7: Reportes
│   ├── core/
│   │   ├── graph/
│   │   │   └── state_graph.py    # Orquestación del flujo
│   │   └── orchestrator/
│   │       └── orchestrator.py   # Orquestador principal
│   └── config.py                 # Configuración centralizada
├── sql/
│   ├── init.sql                  # Inicialización de BD
│   └── resetdb.py               # Reset de BD
├── results/                      # Resultados generados
├── main.py                      # Punto de entrada
├── requirements.txt             # Dependencias
├── docker-compose.yml           # Docker para PostgreSQL
└── README.md                    # Este archivo
```

## 🔄 Flujo de Procesamiento

### Modos de Operación

El sistema puede operar en dos modos diferentes según la configuración de `EXTRACT_FROM_OCR`:

#### **Modo OCR Directo** (`EXTRACT_FROM_OCR=true`)
```
PASO 1: OCR → PASO 2: Versionado → PASO 3: Clasificación → PASO 4: Vectorización (omitida) → PASO 5: Comparación → PASO 6: Legalización → PASO 7: Reporte (desde OCR)
```

#### **Modo Sin Vectorización** (`EXTRACT_FROM_OCR=false`)
```
PASO 1: OCR → PASO 2: Versionado → PASO 3: Clasificación → PASO 4: Vectorización (omitida) → PASO 5: Comparación → PASO 6: Legalización → PASO 7: Reporte (desde OCR)
```

### PASO 1: OCR (OCRAgent)
**Objetivo**: Convertir documentos PDF a texto usando Azure Document Intelligence

**Entrada**: Archivos PDF en `data/{empresa}/`
**Salida**: Lista de documentos con texto extraído
```json
[
  {
    "filename": "documento.pdf",
    "path": "data/empresa/documento.pdf", 
    "text": "Texto extraído del documento..."
  }
]
```

**Características**:
- Limpieza automática del texto OCR
- Eliminación de caracteres especiales y ruido
- Normalización de espacios y saltos de línea

### PASO 2: Extracción de Fechas (DateExtractorAgent)
**Objetivo**: Extraer y normalizar todas las fechas relevantes

**Entrada**: Lista de documentos del PASO 1
**Salida**: Lista con fechas normalizadas
```json
[
  {
    "filename": "documento.pdf",
    "fechas": ["2020-01-15", "2020-06-22"]
  }
]
```

**Características**:
- Detección de múltiples formatos de fecha
- Normalización a formato YYYY-MM-DD
- Identificación de fechas más relevantes



### PASO 3: Clasificación de Documentos (DocumentClassifierAgent)
**Objetivo**: Clasificar documentos por tipo formal

**Entrada**: Documentos con texto, fechas y tipo de empresa
**Salida**: Documentos clasificados
```json
[
  {
    "filename": "documento.pdf",
    "fecha": "2020-01-15",
    "clasificacion": "escritura_publica"
  }
]
```

**Tipos de clasificación**:
- `escritura_publica`: Escrituras de constitución/modificación
- `inscripcion_cbr`: Inscripciones en registro de comercio
- `publicacion_diario_oficial`: Publicaciones en DO

### PASO 4: Vectorización (VectorizationAgent)
**Objetivo**: Dividir documentos en chunks y generar embeddings

**Entrada**: Documentos clasificados
**Salida**: Chunks vectorizados en PostgreSQL (vectorización omitida en ambos modos)

**Comportamiento según configuración**:

**Modo OCR Directo** (`EXTRACT_FROM_OCR=true`):
```json
{
  "collection": "legal_empresa",
  "documentos_procesados": 0,
  "total_chunks": 0,
  "modo": "OCR_DIRECTO",
  "mensaje": "Vectorización omitida - usando extracción directa desde OCR"
}
```

**Modo Sin Vectorización** (`EXTRACT_FROM_OCR=false`):
```json
{
  "collection": "legal_empresa",
  "documentos_procesados": 0,
  "total_chunks": 0,
  "modo": "SIN_VECTORIZACION",
  "mensaje": "Vectorización omitida - usando extracción directa desde OCR"
}
```

**Características**:
- **OCR Directo**: Vectorización omitida para mayor velocidad
- **Sin Vectorización**: Vectorización omitida para mayor velocidad
- No requiere PostgreSQL ni embeddings
- Procesamiento directo desde documentos OCR
- Prevención de duplicados

### PASO 5: Versionado (VersioningAgent)
**Objetivo**: Asignar versiones cronológicas a documentos

**Entrada**: Documentos clasificados con fechas
**Salida**: Documentos versionados
```json
{
  "escritura_publica": [
    {
      "filename": "constitución.pdf",
      "fecha": "2020-01-15",
      "version": 1,
      "clasificacion": "escritura_publica"
    }
  ]
}
```

**Características**:
- Versionado por tipo de documento
- Ordenamiento cronológico
- Identificación de documentos base

### PASO 6: Comparación de Versiones (VersionComparerAgent)
**Objetivo**: Comparar versiones y extraer cambios legales

**Entrada**: Documentos versionados
**Salida**: Comparaciones y cambios detectados
```json
{
  "escritura_publica": [
    {
      "de": 1,
      "a": 2,
      "archivo_v1": "constitución.pdf",
      "archivo_vn": "modificación.pdf",
      "fecha_v1": "2020-01-15",
      "fecha_vn": "2022-06-15",
      "cambios": ["Cambio de razón social", "Modificación de capital"],
      "resumen": "Se modificó la razón social y se aumentó el capital"
    }
  ]
}
```

**Características**:
- Comparación semántica de contenido
- Detección automática de cambios
- Resumen de modificaciones

### PASO 7: Generación de Reportes (ReportGeneratorAgent)
**Objetivo**: Generar reporte legal completo y estructurado

**Entrada**: Todos los resultados anteriores
**Salida**: Reporte legal completo

**Modos de extracción**:
- **OCR Directo** (`EXTRACT_FROM_OCR=true`): Extrae información directamente desde documentos OCR
  - Más rápido y eficiente
  - No requiere vectorización previa
  - Recomendado para pruebas y desarrollo

- **Sin Vectorización** (`EXTRACT_FROM_OCR=false`): Extrae información directamente desde documentos OCR (sin vectorizar)
  - Más rápido y eficiente
  - No requiere PostgreSQL ni vectorización
  - Recomendado cuando no se necesita búsqueda semántica
```json
{
  "encabezado": {
    "razon_social": "EMPRESA SPA",
    "rut": "76.154.106-4",
    "nombre_fantasia": "EMPRESA SPA"
  },
  "constitucion": {
    "tipo_sociedad": "SpA",
    "domicilio": "Santiago, Chile",
    "objeto_social": "Actividades comerciales",
    "fecha_constitucion": "15-01-2020"
  },
  "capital_social": {
    "capital_suscrito": "1.000.000",
    "capital_pagado": "500.000",
    "responsabilidad_socios": "Limitada"
  },
  "administracion": {
    "tipo_administracion": "Unipersonal",
    "duracion": "Indefinida"
  },
  "legalizacion": {
    "escritura_constitucion": {
      "repertorio": "4060-2011",
      "notaria": "Vigésimo Segunda Notaría",
      "inscripcion_registro_comercio": "27009"
    }
  },
  "poderes_personarias": {
    "facultades_encontradas": [
      {
        "codigo": "72",
        "nombre": "Pago SII",
        "documento": "CONSTITUCION.pdf"
      }
    ]
  },
  "restricciones": {
    "restricciones": []
  }
}
```

**Secciones del reporte**:
- **Encabezado**: Información básica de la empresa
- **Constitución**: Datos de constitución y objeto social
- **Capital Social**: Información financiera
- **Administración**: Estructura administrativa
- **Legalización**: Datos de notaría e inscripción
- **Poderes y Personarias**: Facultades delegadas
- **Restricciones**: Limitaciones legales

## 🚀 Instalación y Configuración

### Prerrequisitos
- Python 3.8+
- Docker y Docker Compose
- Cuenta de Azure con Document Intelligence y OpenAI

### 1. Clonar y configurar
```bash
git clone <repository>
cd Poc-Legales
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Configurar variables de entorno
```bash
cp env.example .env
```

Editar `.env` con tus credenciales:
```env
# Azure Document Intelligence
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-key

# Azure OpenAI
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# PostgreSQL
POSTGRES_CONNECTION_STRING=postgresql://user:password@localhost:5432/legal_docs

# Configuración de extracción
EXTRACT_FROM_OCR=true  # true = OCR directo, false = sin vectorización
```

### 3. Inicializar base de datos
```bash
docker-compose up -d
python sql/resetdb.py
```

### 4. Preparar documentos
Colocar documentos PDF en `data/{nombre_empresa}/`

## 📋 Uso

### Procesar una empresa
```bash
python main.py "NOMBRE EMPRESA"
```

### Ejemplo
```bash
python main.py "BASTIAS SPA 76.154.106-4"
```

### Resultados
Los resultados se guardan en `results/`:
- `{empresa}_ocr_results.json`: Texto extraído
- `{empresa}_date_results.json`: Fechas extraídas

- `{empresa}_classification_results.json`: Documentos clasificados
- `{empresa}_vectorization_results.json`: Vectorización
- `{empresa}_versioning_results.json`: Versionado
- `{empresa}_comparison_results.json`: Comparaciones
- `{empresa}_report_results.json`: Reporte final

## 🔧 Configuración Avanzada

### Agentes Personalizables
Cada agente puede ser configurado independientemente:

- **Prompts**: Modificar prompts en cada agente
- **Modelos**: Cambiar modelos de Azure OpenAI
- **Parámetros**: Ajustar parámetros de procesamiento

### Base de Datos
- **Esquema**: Modificar `sql/init.sql`
- **Facultades**: Tabla `langchain_pg_facultades` para códigos de facultades
- **Vectores**: Tabla `langchain_pg_embedding` para embeddings

### Docker
```bash
# Iniciar PostgreSQL
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener
docker-compose down
```

## 🐛 Troubleshooting

### Problemas Comunes

**Error de conexión a Azure**
- Verificar credenciales en `.env`
- Comprobar endpoints y versiones de API

**Error de base de datos**
- Verificar que PostgreSQL esté corriendo
- Ejecutar `python sql/resetdb.py`

**Error de parsing JSON**
- El sistema maneja automáticamente JSON malformado
- Verificar prompts en los agentes

**Documentos no procesados**
- Verificar formato PDF
- Comprobar permisos de archivos

### Logs
Los logs se muestran en consola con información detallada de cada paso.

## 📊 Rendimiento

### Optimizaciones
- **Vectorización incremental**: Solo procesa documentos nuevos
- **Chunking inteligente**: División semántica optimizada
- **Caché de embeddings**: Evita reprocesamiento

### Escalabilidad
- **Procesamiento paralelo**: Múltiples documentos simultáneos
- **Base de datos optimizada**: Índices para búsquedas rápidas
- **Gestión de memoria**: Procesamiento por lotes

## 🤝 Contribución

1. Fork el proyecto
2. Crear rama feature (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 📞 Soporte

Para soporte técnico o preguntas:
- Crear un issue en GitHub
- Contactar al equipo de desarrollo

---

**Poc Legales** - Sistema automatizado de procesamiento de documentos legales 