# SDD-010 — Diseño

## 1. El componente

```python
class EmptyState(QWidget):
    """Mensaje centrado de «aquí no hay nada todavía»."""

    def __init__(self, titulo="", detalle="", siguiente="", parent=None):
        # VBoxLayout centrado, sin márgenes: el texto es el protagonista.
        #   QLabel[titulo]   → role="titulo"
        #   QLabel[detalle]  → role="suave", con salto de línea
        #   QLabel[siguiente]→ role="seccion"

    def set_text(self, titulo, detalle="", siguiente=""): ...
    def mostrar(self, visible: bool): ...
```

Centro vertical: se mete en un layout con `addStretch()` antes y después, o
con `addStretch(1)`; centrar con `Qt.AlignCenter` sobre toda la zona fuerza a
la zona a crecer y descuadra el scroll.

## 2. Cómo se enchufa en cada pestaña

El patrón es el mismo en las cinco:

```python
self.vacio = EmptyState(...)
layout.insertWidget(0, self.vacio)      # antes del contenido
self._alternar_vacio()                  # desde refresh()
```

y el helper compartido decide:

```python
def alternar_estado_vacio(estado, widgets, vacio):
    estado.mostrar(vacio)
    for w in widgets:
        w.setVisible(not vacio)
```

Se comprueba el contenido antes de decidir, no la bandera `vacio` a secas:
una tabla con filas nunca se oculta, aunque el mensaje siga puesto.

| Pestaña | Se oculta cuando no hay… | Se oculta siempre |
|---|---|---|
| Datos | `table`, `table_perfil` | — |
| Preprocesamiento | `grupos` (el acordeón) | `btn_apply` nunca |
| Entrenamiento | el formulario de la página «Modelo» | el acordeón y «Entrenar» |
| Resultados | `tbl_metricas`, `canvas` | — |
| Predicción | `table_entrada`, `table_salida` | — |

Preprocesamiento y Entrenamiento son los casos delicados: el botón de acción
se queda fuera del alternador (R-007), porque «Aplicar preprocesamiento» y
«Entrenar modelo» tienen que seguir en su sitio aunque el contenido esté
oculto; si no, los tests de geometría y el usuario se quedan sin la acción
principal.

## 3. Los textos

Se nombran por lo que falta y por el paso que lo resuelve:

| Pestaña | Título | Siguiente paso |
|---|---|---|
| Datos | «No hay ningún archivo cargado» | «Carga un CSV o un Excel en el paso 1» |
| Preprocesamiento | «No hay datos que limpiar» | «Vuelve al paso 1 y carga un archivo» |
| Entrenamiento | «Faltan los datos ya limpiados» | «Aplica el preprocesamiento en el paso 2» |
| Resultados | «Todavía no hay un modelo entrenado» | «Entrena un modelo en el paso 3» |
| Predicción | «No hay ningún modelo cargado» | «Carga un archivo .automl, o entrena uno» |

«Vuelve al paso 1» en vez de «carga un archivo»: la barra lateral ya existe y
el mensaje debe concordar con ella, no inventar otra forma de expresarlo.

## 4. Por qué sin botón

Un `QPushButton` dentro del estado vacío que salte a otra pestaña es
exactamente el acoplamiento que la spec 008 eliminó: el botón no puede tocar
el `QStackedWidget` del padre. Para que funcionara habría que emitir otra
señal y duplicar la navegación que ya existe. El texto dice qué hacer y la
barra lateral lo hace.

## 5. Alternativas descartadas

| Alternativa | Por qué no |
|---|---|
| Icono grande en el hueco | Duplica el escudo de la lateral y añade un PNG más |
| Botón de acción en el mensaje | Obliga a reintroducir la navegación por widget |
| Un único mensaje reutilizado en las cinco | El usuario necesita saber **qué** falta en esta pestaña, no un texto genérico |
| Borrar las tablas del layout | Los tests las buscan por atributo y desaparecerían (R-008) |