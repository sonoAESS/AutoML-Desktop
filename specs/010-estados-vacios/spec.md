# SDD-010 — Estados vacíos explicados

| Campo | Valor |
|---|---|
| Estado | Especificación — **implementada** |
| Orden de ejecución | 4 de 4 (de la rama `design/redesign-ui`) |
| Ámbito | `ui/empty_state.py` (nuevo), las cinco pestañas, `tests/test_empty_state_ui.py` |
| Depende de | SDD-007 (tokens), SDD-008 (estructura), SDD-009 (estados) |
| Bloquea a | — |

## 1. Problema

La spec 009 apagó los controles que no aplican, pero dejó el hueco que
había al lado. Cuando no hay nada que mostrar, la pestaña enseña una tabla
vacía con su cabecera y varios cientos de píxeles de blanco:

| Pestaña | Qué se ve sin datos |
|---|---|
| Resultados | «Sin resultados todavía.» y una tabla de métricas vacía |
| Predicción | «Ningún archivo cargado.» y dos tablas vacías |
| Datos | Dos tablas vacías con cabecera |
| Preprocesamiento | Cinco grupos con controles apagados |
| Entrenamiento | Formulario con combos vacíos y un texto de aviso |

Es el peor de los dos estados posibles: la interfaz **parece** rota en vez de
parecer vacía. Un texto pequeño arriba del todo no se lee, y la tabla vacía
no dice nada.

## 2. Objetivo

Que cada pestaña sin datos diga **qué falta** y **qué hacer a continuación**,
en el centro de la zona de contenido, en vez de mostrar una tabla vacía que
ocupa media pantalla.

## 3. Alcance

### 3.1 Incluido

- `ui/empty_state.py` con un componente reutilizable `EmptyState`.
- Alternancia entre contenido y estado vacío en las cinco pestañas.
- Mensajes que nombran el paso siguiente, no solo la carencia.

### 3.2 No incluido

- Iconos o ilustraciones en los estados vacíos: el escudo ya está en la
  barra lateral y un segundo logo en cada hueco duplicaría el ruido.
- Botones dentro del estado vacío que disparen acciones: el usuario ya tiene
  la barra lateral y los botones de la propia pestaña.
- Internationalización: los mensajes siguen en español, como el resto.

## 4. Requisitos

- **R-001.** `ui/empty_state.py::EmptyState(QWidget)` con tres niveles de
  texto: **título** (qué falta), **detalle** (por qué) y **siguiente paso**
  (qué hacer). Los tres son opcionales menos el título.
- **R-002.** El contenido va centrado horizontal y verticalmente en la zona
  que ocupa, con los roles `titulo`, `suave` y `seccion` de la hoja de estilo.
- **R-003.** `mostrar(con_vacio: bool)` deja el estado vacío visible u oculto
  sin recrearlo, para no perder los textos que ya tenga.
- **R-004.** `alternar_estado_vacio(estado, widgets, vacio)`: la función que
  cada pestaña usa para mostrar el mensaje y ocultar a la vez el contenido
  vacío, o al revés. Oculta los widgets **solo** cuando están vacíos de
  verdad, nunca cuando tienen filas.
- **R-005.** El contenido con datos no se oculta nunca: un dataset con seis
  filas enseña su tabla y no un mensaje.
- **R-006.** Los cinco mensajes nombian el paso siguiente con su número, para
  que conecten con la barra lateral.
- **R-007.** El estado vacío de Preprocesamiento no oculta el botón «Aplicar
  preprocesamiento»: los tests de geometría lo verifican y, sobre todo, es la
  acción que cierra el paso.
- **R-008.** Los widgets con contenido se ocultan con `setVisible(False)`, no
  borrándose de los `layout`: los tests de UI los localizan por atributo.

## 5. Invariantes

| Invariante | Origen |
|---|---|
| `core/` no se toca: los mensajes son de presentación | AGENTS §7.4 |
| Ningún color literal: el componente usa los roles de la hoja de estilo | AGENTS §3.6 |
| Sin dependencias nuevas | AGENTS §7.3 |
| Mensajes en español | AGENTS §4 |
| Los 393 tests existentes siguen en verde | AGENTS §8 |

## 6. Criterios de aceptación

- **C-001.** `EmptyState` muestra título, detalle y siguiente paso, y es
  idempotente al llamar a `mostrar` dos veces.
- **C-002.** Sin datos, las cinco pestañas muestran su estado vacío.
- **C-003.** Con datos, ninguna pestaña muestra el mensaje y sí su contenido.
- **C-004.** El mensaje nombra el número del paso siguiente.
- **C-005.** «Aplicar preprocesamiento» sigue visible sin datos.
- **C-006.** Los widgets con contenido existen siempre como atributos, aunque
  estén ocultos: la suite existente los busca por nombre.
- **C-007.** Ninguna tabla se queda visible y vacía a la vez que el mensaje.

## 7. Riesgos

| Riesgo | Mitigación |
|---|---|
| Ocultar widgets rompe los tests que los buscan | Solo se usa `setVisible`; los atributos y el layout no cambian (R-008) |
| Un mensaje que se equivoca en el paso siguiente | El test comprueba el número contra `PASOS` de la barra lateral |
| El estado vacío tapa algo útil | Nunca se oculta un widget con contenido (R-005) |