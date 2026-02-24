# Guía de Sincronización y Export de Feedbacks

## 🎯 Problema Resuelto

Tus feedbacks se guardan automáticamente en GitHub (repo `lm-lab-feedbacks`), pero **el backend en Render se resetea** cada vez que:
- Haces un nuevo deploy
- Render reinicia el servidor
- Cambias el plan de Render

Esta guía explica cómo **recuperar** y **descargar** todos tus feedbacks.

---

## 📥 Método 1: Sincronizar desde GitHub (Recomendado)

### Usar botón "Sync from GitHub" en el viewer

1. **Abrir viewer:** `https://lm-lab.onrender.com/api/v1/feedback/viewer`
2. **Click en "Sync from GitHub"** (botón superior con icono de refresh)
3. **Esperar confirmación:** Se descargará todo desde GitHub al backend local
4. **Resultado:** Todos los feedbacks visibles en el viewer

**Cuándo usar:**
- ✅ Después de cada deploy nuevo en Render
- ✅ Si el backend se reinició y los feedbacks desaparecieron
- ✅ Para recuperar feedbacks antiguos

**Endpoint API:**
```bash
POST https://lm-lab.onrender.com/api/v1/feedback/sync-from-github
```

**Response:**
```json
{
  "ok": true,
  "synced": 42,
  "total": 42,
  "errors": []
}
```

---

## 💾 Método 2: Descargar ZIP completo

### Usar botón "Export ZIP" en el viewer

1. **Abrir viewer:** `https://lm-lab.onrender.com/api/v1/feedback/viewer`
2. **Click en "Export ZIP"** (botón superior con icono de descarga)
3. **Guardar archivo:** Se descarga `feedbacks_export_YYYYMMDD_HHMMSS.zip`

**Contenido del ZIP:**
```
feedbacks_export_20260224_211500.zip/
├── active/
│   ├── ngram/
│   │   ├── general/
│   │   │   ├── 20260224T201700Z.json
│   │   │   └── 20260224T201700Z_user.jpg
│   │   └── smoothing/
│   │       └── 20260224T202300Z.json
│   └── mlp/
│       └── general/
│           └── 20260224T203000Z.json
└── archived/
    └── ngram/
        └── general/
            └── 20260220T150000Z.json
```

**Endpoint API:**
```bash
GET https://lm-lab.onrender.com/api/v1/feedback/export
```

**Response:** Binary ZIP file

---

## 📂 Método 3: Clonar repo GitHub directamente

### Para tener backup local permanente

```bash
# Clone el repo
git clone https://github.com/adrilaynez/lm-lab-feedbacks.git

# Ver feedbacks
cd lm-lab-feedbacks/data/feedback/
ls -R

# Pulls periódicos para actualizar
git pull
```

**Ventajas:**
- ✅ Backup local permanente en tu ordenador
- ✅ Sincronización automática con `git pull`
- ✅ Historial completo de cambios (Git)
- ✅ No depende del backend

---

## 🔄 Workflow Recomendado

### Flujo diario
1. **Usuarios envían feedback** → Auto-commit a GitHub ✅
2. **Ver feedbacks en viewer** → Click "Sync" si es nuevo deploy
3. **Trabajo semanal:**
   - Clasificar feedbacks (estado, tags, owner)
   - Crear issues en GitHub
   - Marcar como fixed/archived

### Backup semanal
```bash
# En tu ordenador
cd ~/backups/
git clone https://github.com/adrilaynez/lm-lab-feedbacks.git

# Cada semana
cd lm-lab-feedbacks
git pull
```

**Opcional:** Automatizar con cron job:
```bash
# Ejecutar cada lunes a las 9am
0 9 * * 1 cd ~/backups/lm-lab-feedbacks && git pull
```

---

## 🚀 Nuevas Features del Viewer

### Botones de Sincronización

#### Sync from GitHub
- **Icono:** ↻ (refresh)
- **Función:** Pull todos los feedbacks desde GitHub repo
- **Animación:** Spinner mientras sincroniza
- **Resultado:** Alert con `✅ Synced X/Y files`

#### Export ZIP
- **Icono:** ↓ (download)
- **Función:** Descargar todos los feedbacks como ZIP
- **Resultado:** Archivo `feedbacks_export_YYYYMMDD_HHMMSS.zip`

### Filtros Avanzados

#### State Filter
- **new** 🔵 - Recién recibido
- **triaged** 🟡 - Revisado/clasificado
- **in_progress** 🟣 - En desarrollo
- **fixed** 🟢 - Resuelto
- **wont_fix** 🔴 - No se implementará
- **archived** ⚪ - Archivado

#### User Filter
- Muestra todos los usuarios únicos
- Formato: `Name (count)` o `anon abc12345… (count)`
- Filtra por name si existe, sino por anon_id

#### Unread Filter
- **Show: all** - Todos los feedbacks
- **Show: unread only** - Solo no leídos
- **Show: read only** - Solo leídos

### Badges Visuales

#### State Badge
```html
<span class="state-badge state-new">NEW</span>
<span class="state-badge state-fixed">FIXED</span>
```

#### Tags Badges
```html
<span class="tag-badge">bug</span>
<span class="tag-badge">ui</span>
<span class="tag-badge">+3</span>  <!-- Si hay más de 3 tags -->
```

#### Owner Badge
```html
<span class="owner-badge">👤 adrilaynez</span>
```

### Metadata Extendida (en card detail)

Ahora se muestra:
- **Type:** Bug/Idea/Question
- **URL:** Link clickeable a la página exacta
- **Viewport:** `1920×1080`
- **Theme:** dark/light
- **Lang:** es/en
- **UA:** User Agent (hover para ver completo)

### Botones de Screenshot

Cada screenshot ahora tiene:
- **Open** - Abrir en nueva pestaña
- **Download** - Descargar directamente

---

## 🔧 Troubleshooting

### "Sync failed: GitHub API error"

**Causa:** Token inválido o expirado

**Solución:**
1. Generar nuevo token: https://github.com/settings/tokens
2. Scope: `repo` (full control)
3. Actualizar en Render: `GITHUB_TOKEN=ghp_xxxxx`
4. Restart servicio

### "Export ZIP está vacío"

**Causa:** No hay feedbacks en el backend local

**Solución:**
1. Click "Sync from GitHub" primero
2. Luego "Export ZIP"

### "Feedbacks desaparecieron tras deploy"

**Causa:** Filesystem efímero de Render

**Solución:**
1. Click "Sync from GitHub"
2. Todos los feedbacks vuelven desde GitHub repo

---

## 📊 Estadísticas y Analytics

### Ver contadores en viewer
- **Total count:** `42 active · 12 archived`
- **Unread count:** `Unread: 5`
- **Por estado:** Filtrar y contar visualmente

### Query manual (opcional)

```bash
# Contar feedbacks por estado
cd lm-lab-feedbacks/data/feedback
grep -r '"state"' . | grep -o 'state.*' | sort | uniq -c

# Contar por página
ls -1 data/feedback/ | while read page; do
  count=$(find data/feedback/$page -name "*.json" | wc -l)
  echo "$page: $count"
done
```

---

## 🎯 Mejoras Futuras Propuestas

### A) Auto-sync on startup
Añadir en backend startup:
```python
@app.on_event("startup")
async def startup_event():
    # Auto-sync feedbacks desde GitHub al iniciar
    if FEEDBACK_AUTO_SYNC:
        asyncio.create_task(sync_feedbacks_from_github())
```

### B) Webhooks de GitHub
Sincronización bidireccional:
- Push desde backend → GitHub ✅ (ya implementado)
- Push desde GitHub → Backend (via webhook)

### C) Scheduled backups
Cron job en backend para export automático:
```python
# Cada lunes a las 3am UTC
@app.on_event("startup")
async def schedule_backup():
    scheduler = BackgroundScheduler()
    scheduler.add_job(export_to_s3, 'cron', day_of_week='mon', hour=3)
    scheduler.start()
```

### D) Dashboard de Analytics
- Gráficas de feedbacks por día/semana/mes
- Tags más usados (word cloud)
- Time-to-fix average
- Ratio fixed/wont_fix

---

**Última actualización:** 2026-02-24  
**Estado:** ✅ Totalmente funcional
