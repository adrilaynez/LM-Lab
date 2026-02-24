# Feedback Persistence Setup

## Problema en Render
Render usa **filesystem efímero**: cualquier archivo creado en `data/feedback/` se **pierde al reiniciar el servicio**. 

## Solución: Auto-commit a GitHub
El backend guarda cada feedback automáticamente en un **repositorio GitHub** dedicado, garantizando persistencia permanente.

---

## Setup paso a paso

### 1. Crear repositorio de feedbacks
```bash
# En tu cuenta GitHub, crea un repo (puede ser privado):
# Nombre sugerido: lm-lab-feedbacks
# Descripción: "User feedback storage for LM-Lab"
```

### 2. Generar GitHub Personal Access Token (PAT)

1. Ve a: https://github.com/settings/tokens
2. Click **"Generate new token (classic)"**
3. Nombre: `LM-Lab Feedback Writer`
4. Scopes necesarios:
   - ✅ `repo` (full control of private repositories)
     - Incluye: `repo:status`, `repo_deployment`, `public_repo`, `repo:invite`
5. Click **"Generate token"**
6. **Copia el token** (solo se muestra una vez): `ghp_xxxxxxxxxxxxxxxxxxxx`

### 3. Configurar variables de entorno en Render

En tu servicio de Render (https://dashboard.render.com):

1. Ve a tu servicio `lm-lab`
2. Tab **"Environment"**
3. Añade estas variables:

```bash
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx    # Tu PAT del paso 2
FEEDBACK_REPO_OWNER=adrilaynez            # Tu usuario GitHub
FEEDBACK_REPO_NAME=lm-lab-feedbacks       # Nombre del repo creado
FEEDBACK_AUTO_COMMIT=true                 # Activar auto-commit
```

4. Click **"Save Changes"** → el servicio se reiniciará automáticamente

### 4. Verificar funcionamiento

1. Envía un feedback desde tu web (https://adrianlaynez.dev)
2. Verifica en GitHub que apareció:
   ```
   https://github.com/adrilaynez/lm-lab-feedbacks/tree/main/data/feedback
   ```
3. Cada feedback crea un commit automático con mensaje:
   ```
   [Feedback] <page>/<section>: <title>
   ```

---

## Configuración opcional

### Rama personalizada
Por defecto usa `main`. Para cambiar:
```bash
FEEDBACK_REPO_BRANCH=feedbacks   # Usar rama 'feedbacks'
```

### Desactivar auto-commit (solo local)
```bash
FEEDBACK_AUTO_COMMIT=false   # Solo guardar en filesystem local
```

---

## Recuperar feedbacks antiguos

Si ya tienes feedbacks en el filesystem de Render antes de configurar esto:

1. Descarga manualmente la carpeta `data/feedback/` desde el viewer
2. Sube los archivos al repo GitHub:
   ```bash
   git clone https://github.com/adrilaynez/lm-lab-feedbacks.git
   cd lm-lab-feedbacks
   # Copia data/feedback/* aquí
   git add .
   git commit -m "Import existing feedbacks"
   git push
   ```

---

## Troubleshooting

### Error: "GITHUB_TOKEN is not configured"
- Verifica que `GITHUB_TOKEN` esté en las variables de entorno de Render
- Reinicia el servicio después de añadir la variable

### Error: "Failed to commit feedback to GitHub"
- Verifica que el token tenga permisos `repo`
- Verifica que `FEEDBACK_REPO_OWNER` y `FEEDBACK_REPO_NAME` sean correctos
- Revisa los logs de Render para ver el error específico

### Los commits no aparecen en GitHub
- Verifica que el repo exista y sea accesible
- Verifica que la rama (`main` o custom) exista
- Prueba crear un commit manual en el repo para verificar permisos

---

## Creación de GitHub Issues

Para crear issues desde el viewer también necesitas `GITHUB_TOKEN` (usa el mismo):

```bash
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
```

El viewer permite crear issues en cualquier repo (configurable en la UI), pero el token debe tener permisos `repo` en ese repo.
