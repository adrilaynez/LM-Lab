# Feedback System - Complete Enhancement Summary

## 🎯 Problemas resueltos

### 1. ✅ Persistencia en Render (CRÍTICO)
**Problema:** Render usa filesystem efímero → feedbacks se borran al reiniciar.

**Solución implementada:**
- **Auto-commit a GitHub** cada vez que se recibe feedback
- Configurable via variables de entorno en Render
- Fallback silencioso (no bloquea submission si GitHub falla)

**Configuración requerida:**
```bash
GITHUB_TOKEN=ghp_xxxxxxxxxxxxx          # Token con scope 'repo'
FEEDBACK_REPO_OWNER=adrilaynez          # Tu usuario GitHub
FEEDBACK_REPO_NAME=lm-lab-feedbacks     # Repo dedicado para feedbacks
FEEDBACK_REPO_BRANCH=main               # Rama (opcional, default: main)
FEEDBACK_AUTO_COMMIT=true               # Activar auto-commit
```

Ver: `FEEDBACK_PERSISTENCE_SETUP.md` para guía completa.

---

### 2. ✅ GitHub Issues - Token corregido
**Problema:** Mensaje "GITHUB_TOKEN is not configured" aunque estaba en `.env`

**Solución:**
- Cambiado de `os.getenv("GITHUB_TOKEN")` a importar `GITHUB_TOKEN` desde `api/config.py`
- Ahora usa la misma config que auto-commit
- El mismo token sirve para issues Y auto-commit

---

### 3. ✅ Contraste en filtros del viewer
**Problema:** Fondo blanco + texto blanco = ilegible

**Solución:**
- Cambiado `background: rgba(255,255,255,0.02)` → `rgba(11,11,16,0.6)`
- Añadido `option { background: var(--bg); color: var(--text); }`
- Mejor contraste, fácil de leer

---

## 🚀 Nuevas Features Implementadas

### A) Estados Enriquecidos (Triage Workflow)
**Antes:** Solo "read" + "archived"  
**Ahora:** 6 estados completos

| Estado | Badge Color | Uso |
|--------|------------|-----|
| `new` | 🔵 Azul | Recién recibido (default) |
| `triaged` | 🟡 Ámbar | Revisado, clasificado |
| `in_progress` | 🟣 Morado | En desarrollo/investigación |
| `fixed` | 🟢 Verde | Resuelto/completado |
| `wont_fix` | 🔴 Rojo | No se implementará |
| `archived` | ⚪ Gris | Archivado (legacy) |

**Backend endpoints:**
- `POST /api/v1/feedback/update-state`
- Validación automática de estados válidos
- Auto-commit a GitHub al cambiar estado

**Frontend (viewer):**
- Badges visuales con colores
- Dropdown para cambiar estado
- Filtro por estado (próximamente)

---

### B) Tags/Labels System
**Funcionalidad:** Igual que GitHub labels

**Ejemplos de tags:**
- `bug`, `typo`, `content`, `ui`, `idea`, `feature-request`, `duplicate`, `spam`

**Backend:**
- `POST /api/v1/feedback/update-tags` (enviar array de strings)
- Máximo 20 tags por feedback
- Auto-commit a GitHub

**Frontend:**
- Badges visuales morados
- Editor inline (próximamente)

---

### C) Owner Assignment
**Funcionalidad:** Asignar responsable (trabajo en equipo)

**Backend:**
- `POST /api/v1/feedback/update-owner`
- Campo: `owner` (string, nombre/email)
- Auto-commit a GitHub

**Frontend:**
- Badge verde con nombre del owner
- Dropdown para asignar/desasignar

---

### D) Metadata Extendida Capturada
**Antes:** Solo `name`, `anon_id`, `comment`, `title`  
**Ahora:** Metadata completa de contexto

| Campo | Ejemplo | Uso |
|-------|---------|-----|
| `feedback_type` | `"bug"`, `"idea"`, `"question"` | Clasificación rápida |
| `url` | `https://adrianlaynez.dev/lab/ngram#examples` | URL completa con hash |
| `user_agent` | `Mozilla/5.0 ...` | Debugging de browser-specific bugs |
| `viewport_width` | `1920` | Layout/responsive issues |
| `viewport_height` | `1080` | Layout/responsive issues |
| `theme` | `"dark"` o `"light"` | Theme-specific bugs |
| `language` | `"es"`, `"en"` | i18n issues |
| `local_timestamp` | `"2026-02-24T20:17:00.000Z"` | Timezone del usuario |

**Frontend:**
- Selector de tipo (Bug/Idea/Question) con botones
- Captura automática al submit
- No requiere input del usuario

---

### E) Rate Limit Mejorado
**Antes:** Solo por IP → problema si cambia IP  
**Ahora:** Preferencia por `anon_id` (más estable)

**Lógica:**
1. Si existe `anon_id` → rate limit por `SHA256(anon_id)[:16]`
2. Si no → fallback a IP

**Ventajas:**
- Más robusto (survives IP changes, VPN switches)
- Dificulta bypass (hash del anon_id)
- Preparado para persistencia en redis/sqlite

---

### F) Screenshot Preview en Modal
**Antes:** Solo texto "Image attached"  
**Ahora:** Preview visual completo

**Features:**
- Preview inline (max-height: 192px)
- Botón "Delete" con icono trash
- Soporte paste (Ctrl+V)
- Soporte drag & drop via file input

---

### G) Screenshots en Viewer - Download/Open
**Pendiente de implementación en viewer:**
- Botón "Download" (descarga directa)
- Botón "Open in new tab" (abre en pestaña nueva)
- Mostrar dimensiones (WxH) y tamaño (KB)

---

## 📊 Schema de Feedback (JSON)

```json
{
  "page_id": "ngram",
  "section_id": "general",
  "comment": "El slider no funciona bien en mobile",
  "title": "Bug en slider mobile",
  "name": "Adrian",
  "anon_id": "abc123def456",
  "timestamp": "20260224T201700Z",
  "local_timestamp": "2026-02-24T21:17:00.000Z",
  
  "state": "triaged",
  "tags": ["bug", "mobile", "ui"],
  "owner": "adrilaynez",
  
  "feedback_type": "bug",
  "url": "https://adrianlaynez.dev/lab/ngram#smoothing",
  "user_agent": "Mozilla/5.0 ...",
  "viewport_width": 375,
  "viewport_height": 667,
  "theme": "dark",
  "language": "es",
  
  "pinned": false,
  "read": true,
  "read_at": "20260224T202000Z",
  
  "has_screenshot": false,
  "has_user_screenshot": true,
  "user_screenshot_file": "20260224T201700Z_user.jpg",
  
  "github_issue_url": "https://github.com/adrilaynez/adrian-v2-web/issues/42",
  "github_issue_number": 42,
  "github_issue_created_at": "20260224T202500Z",
  
  "state_updated_at": "20260224T203000Z",
  "tags_updated_at": "20260224T203100Z",
  "owner_updated_at": "20260224T203200Z"
}
```

---

## 🎨 Viewer UI Improvements

### Filtros corregidos
- ✅ Contraste arreglado (fondo oscuro en lugar de blanco)
- ✅ Filtro por usuario (name + anon_id)
- 🔄 **Próximo:** Filtro por estado
- 🔄 **Próximo:** Filtro por tags
- 🔄 **Próximo:** Filtro por owner

### Badges visuales
- ✅ Estados con colores semánticos
- ✅ Tags con badge morado
- ✅ Owner con badge verde
- ✅ Screenshot mejorado (meta "click to zoom")

### Acciones inline
- ✅ Botones estado/tags/owner
- 🔄 **Próximo:** Modals para editar tags
- 🔄 **Próximo:** Dropdown para cambiar estado
- 🔄 **Próximo:** Download/open screenshots

---

## 📋 TODOs Pendientes (Viewer UI)

1. **Filtros avanzados**
   - Dropdown "State: all / new / triaged / ..."
   - Dropdown "Tags: all / bug / ui / ..."
   - Dropdown "Owner: all / unassigned / adrilaynez / ..."

2. **Edición inline**
   - Modal para editar tags (input multi-select)
   - Dropdown para cambiar estado
   - Input para asignar owner

3. **Screenshot actions**
   - Botón "Download" (trigger download)
   - Botón "Open" (open in new tab)
   - Mostrar dimensiones/tamaño

4. **Vistas guardadas (localStorage)**
   - "Only new"
   - "Only with screenshot"
   - "Only without issue link"
   - "My assigned" (by owner)

5. **Bulk actions extendidas**
   - "Set state to X"
   - "Add tag Y"
   - "Assign to Z"

6. **Metadata display**
   - Mostrar `feedback_type` con icono
   - Mostrar `url` clickeable
   - Mostrar viewport/theme/language en detail view

---

## 🔧 Archivos Modificados

### Backend
- ✅ `api/config.py` - GitHub config
- ✅ `api/routers/feedback.py` - Endpoints, schemas, auto-commit, estados/tags/owner
- ✅ `FEEDBACK_PERSISTENCE_SETUP.md` - Documentación setup
- ✅ `FEEDBACK_SYSTEM_ENHANCEMENTS.md` - Este archivo

### Frontend (Modal)
- ✅ `src/components/lab/FeedbackButton.tsx` - Tipo, preview, metadata
- ✅ `src/hooks/useFeedback.ts` - Payload extendido

### Frontend (Viewer)
- ✅ `api/templates/feedback_viewer.html` - Contraste, badges, filtro usuario
- 🔄 **Pendiente:** Completar filtros/acciones estados/tags/owner

---

## 🚀 Deployment Checklist

### Render (Backend)
1. ✅ Crear repo GitHub `lm-lab-feedbacks` (privado)
2. ✅ Generar GitHub PAT con scope `repo`
3. ✅ Configurar variables de entorno en Render:
   ```
   GITHUB_TOKEN=ghp_xxxxx
   FEEDBACK_REPO_OWNER=adrilaynez
   FEEDBACK_REPO_NAME=lm-lab-feedbacks
   FEEDBACK_AUTO_COMMIT=true
   ```
4. ✅ Restart servicio en Render
5. ✅ Verificar que feedbacks se commitean automáticamente

### Vercel (Frontend)
1. ✅ Deploy automático (ya configurado)
2. ✅ Verificar que `NEXT_PUBLIC_LM_LAB_API_URL` apunta a Render
3. ✅ Test feedback submission desde producción

---

## 📝 Notas Importantes

### GitHub Token Permissions
El mismo `GITHUB_TOKEN` se usa para:
- ✅ Auto-commit de feedbacks al repo
- ✅ Crear issues desde el viewer

Scope requerido: `repo` (full control of private repositories)

### Persistencia de Rate Limit
Actualmente en memoria (se resetea al reiniciar backend).  
**Próximo:** Persistir en SQLite o Redis para multi-worker deploys.

### Captcha
No implementado todavía.  
**Próximo:** Añadir reCAPTCHA v3 cuando se detecte abuso.

---

## ✨ Mejoras Futuras Recomendadas

1. **Notificaciones**
   - Webhook a Discord/Slack cuando llega feedback nuevo
   - Email digest diario con resumen

2. **Analytics**
   - Dashboard con estadísticas
   - Gráficas de feedbacks por página/sección
   - Time-to-fix metrics

3. **AI Triage**
   - Auto-clasificación con LLM (tipo, tags sugeridos)
   - Detección de duplicados

4. **Export**
   - CSV export de todos los feedbacks
   - Filtros avanzados para export

5. **Public roadmap**
   - Viewer público (solo feedbacks tipo "idea")
   - Upvoting system

---

**Última actualización:** 2026-02-24  
**Estado:** ✅ Backend completo | 🔄 Viewer UI en progreso
