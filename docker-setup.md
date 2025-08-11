# Configuración de PostgreSQL con pgvector en Docker

## 🐳 Requisitos previos

- Docker Desktop instalado y ejecutándose
- Docker Compose disponible

## 🚀 Pasos para configurar

### 1. Iniciar el contenedor

```bash
# En la raíz del proyecto
docker-compose up -d
```

### 2. Verificar que esté funcionando

```bash
# Ver logs del contenedor
docker-compose logs postgres

# Verificar que esté corriendo
docker ps
```

### 3. Conectar a la base de datos

```bash
# Conectar con psql (si tienes PostgreSQL instalado)
psql -h localhost -p 5432 -U postgres -d postgres

# O usar Docker
docker exec -it poc_legales_postgres psql -U postgres -d postgres
```

### 4. Verificar pgvector

```sql
-- Verificar que pgvector esté habilitado
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Probar funcionalidad de vectores
SELECT '[1,2,3]'::vector;
```

## 🔧 Configuración

### Variables de entorno (ya configuradas en docker-compose.yml):

- **POSTGRES_DB**: postgres
- **POSTGRES_USER**: postgres  
- **POSTGRES_PASSWORD**: 123456
- **Puerto**: 5432

### Conexión desde tu aplicación:

```
postgresql+psycopg2://postgres:123456@localhost:5432/postgres
```

## 🛑 Comandos útiles

```bash
# Detener el contenedor
docker-compose down

# Detener y eliminar volúmenes (¡CUIDADO! Esto borra los datos)
docker-compose down -v

# Reiniciar el contenedor
docker-compose restart

# Ver logs en tiempo real
docker-compose logs -f postgres
```

## 📊 Monitoreo

```bash
# Ver uso de recursos
docker stats poc_legales_postgres

# Ver información del contenedor
docker inspect poc_legales_postgres
```

## 🔍 Troubleshooting

### Si el puerto 5432 está ocupado:
```bash
# Cambiar el puerto en docker-compose.yml
ports:
  - "5433:5432"  # Usar puerto 5433 en lugar de 5432
```

### Si pgvector no se carga:
```bash
# Conectar y ejecutar manualmente
docker exec -it poc_legales_postgres psql -U postgres -d postgres
CREATE EXTENSION IF NOT EXISTS vector;
```

## ✅ Verificación final

Una vez que el contenedor esté corriendo, tu aplicación LangChain debería poder:

1. Conectar a la base de datos
2. Crear automáticamente las tablas necesarias
3. Almacenar y buscar embeddings vectoriales

¡Listo! Tu PostgreSQL con pgvector está configurado y listo para usar con LangChain. 