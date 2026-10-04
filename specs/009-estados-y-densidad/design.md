# SDD-009 — Diseño

## 1. Principio

Una sola regla por pestaña, **derivada de `AppState`**, que decide el estado de
cada control. No se mira nada más: ni la visibilidad de otro widget, ni el
texto de una etiqueta, ni el resultado de una operación anterior. Eso la hace
predecible y testeable.

```python
def _update_enabled_state(self):
    """Habilita cada control según lo que haya en `AppState`.

    Idempotente: leer el estado y volver a aplicarlo no cambia nada.
    """
```

Se invoca desde `refresh()` y **después** de cualquier acción que escriba en
`AppState`. El punto de riesgo es olvidar uno: por eso T-020 exige un test que
recorra todas las transiciones.

## 2. Condiciones por pestaña

### Datos

| Control | Condición |
|---|---|
| `btn_usar_objetivo` | `raw_df is not None` |
| `btn_reperfil` | `raw_df is not None` |
| `table_perfil`, `cb_profile_col` | `raw_df is not None` |

### Preprocesamiento

`hay_datos = raw_df is not None`

| Grupo | Sin datos | Con datos |
|---|---|---|
| `chk_dups` | deshabilitado | habilitado |
| `chk_high_missing`, `cb_num`, `cb_cat` | deshabilitados | habilitados |
| `gb_tipos`, `chk_dates` | deshabilitado | habilitado |
| `table_norm` | deshabilitada (vacía) | habilitada |
| `list_cols` | deshabilitada | habilitada |
| `btn_apply` | deshabilitado | habilitado |

Y la dependencia del perfil semántico, que es lo que hace útil a la
deshabilitación en vez de decorativa:

```python
perfiles = self.state.profiles
hay_numericas = any(p.tipo == "numeric" for p in perfiles)
hay_categoricas = any(p.tipo in {"categorical", "text", "date"} for p in perfiles)
self.cb_num.setEnabled(hay_datos and hay_numericas)
self.cb_cat.setEnabled(hay_datos and hay_categoricas)
```

### Jerarquía (R-005)

Si un grupo está deshabilitado, sus hijos también. Se implementa con una
utilidad, porque repetir el `setEnabled` hijo a hijo en cada pestaña es donde
aparecen los olvidos:

```python
def habilitar_hijos(padre: QWidget, habilitado: bool) -> None:
    for hijo in padre.findChildren(QWidget):
        if hijo is padre or not isinstance(hijo, (QComboBox, QSpinBox, QCheckBox,
                                                  QRadioButton, QLineEdit)):
            continue
        hijo.setEnabled(habilitado)
```

Y el tooltip con el motivo (R-010), que es lo que evita que el usuario piense
que la aplicación está rota:

```python
self.cb_num.setToolTip(
    "No hay variables numéricas en el dataset." if not hay_numericas
    else "Cómo rellenar los valores nulos de las variables numéricas."
)
```

### Entrenamiento

| Control | Condición |
|---|---|
| `btn_train` | `clean_df is not None and target_column and model_name` |
| `gb_tune` y sus hijos | `chk_tune.isChecked()` (ya existe `_toggle_tune_options`) |
| `spn_vecinos` | `cb_balancing` es SMOTE/SMOTEN/SMOTENC (ya existe `_refresh_balancing`) |
| `btn_analizar` | `export_df is not None` y selección activa |

La condición de `btn_train` es la que más valor tiene: hoy se puede pulsar sin
modelo elegido y falla con un `QMessageBox`.

### Resultados

| Control | Condición |
|---|---|
| `btn_save`, `btn_export` | `pipeline is not None` |
| `btn_load_in_predict` | `pipeline is not None` |
| `tbl_metricas`, `canvas` | `pipeline is not None` |
| `gb_comparativa` (interno) | `export_df is not None` y ≥3 modelos (ya implementado) |

### Predicción

| Control | Condición |
|---|---|
| `btn_predict` | `bundle is not None` y hay datos de entrada |
| `btn_export` | `_predictions` no vacío |
| `btn_add_to_data` | `_predictions` no vacío |

## 3. Densidad (R-011 … R-015)

### Textos

| Antes (36–41 car.) | Después | Tooltip |
|---|---|---|
| «Añadir predicciones al dataset» | «Añadir al dataset» | «Añade la columna de predicción al dataset cargado para conservarla al exportar» |
| «Guardar modelo y dataset (.automl)» | «Guardar proyecto» | «Guarda el modelo entrenado, el dataset transformado y los metadatos en un único archivo .automl» |
| «Comparar modelos compatibles» | «Comparar modelos» | «Evalúa todos los modelos compatibles sobre los mismos folds y calcula el test de Friedman» |
| «Volver a detectar tipos» | «Redetectar tipos» | «Vuelve a detectar el tipo de cada columna desde cero, descartando las correcciones manuales» |

El detalle nunca se pierde: va al tooltip. R-011 es un límite de **texto
visible**, no de información.

### Reparto de filas

Regla: `QFormLayout` para etiqueta + control; las acciones en fila propia al
final; ninguna fila de nivel superior con más de 5 widgets. Se verifican con un
test que recorra el árbol de layouts y cuente los `addWidget` de primer nivel.

## 4. Barra de estado (R-016, R-017)

```python
def _estado(self, texto: str, ok: bool = True) -> None:
    self.statusBar().showMessage(texto)
    color = COLORES["suave"] if ok else COLORES["acento"]
    self.statusBar().setStyleSheet(f"color: {color};")
```

No se usa `QMessageBox` para el progreso: los errores **siguen** mostrándose
con `QMessageBox` (AGENTS §5.2); la barra de estado es para el progreso y el
resultado, no un canal de error que se pueda ignorar.

## 5. Alternativas descartadas

| Alternativa | Por qué no |
|---|---|
| Ocultar los controles que no aplican | Cambia el flujo de la app, es una decisión de producto y rompe la familiaridad |
| Un único `_update_enabled_state()` global | Acoplaría las cinco pestañas entre sí; cada una lee `AppState`, que es justamente el punto de `AppState` |
| Deshabilitar solo el botón principal | Es lo que ya hay y no reduce el ruido visual de 13 controles activos |