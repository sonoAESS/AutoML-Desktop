# 011 — Marca visual clara · Tareas

**Spec:** [spec.md](spec.md) · **Diseño:** [design.md](design.md)

---

## Fase 1 · `ui/theme.py`

- [ ] T-001 · Añadir `COLORES["lateral"] = "#24507f"` con su comentario.
- [ ] T-002 · Aclarar `TONOS["sidebar_texto"]` a `#dbe6f3` y
      `TONOS["sidebar_hover"]` a `#1c3c66`.
- [ ] T-003 · `QFrame#lateral` pasa de `_c('marino')` a `_c('lateral')`.
- [ ] T-004 · `escudo_pixmap(px: int = 32)` → `escudo_pixmap(px: int = 64)` y
      actualizar el docstring (el escudo se muestra a 64 px en la barra).
- [ ] T-005 · Eliminar la línea duplicada `QLabel[role="titulo"]`.

## Fase 2 · `ui/sidebar.py`

- [ ] T-006 · Eliminar la etiqueta `organizacion` y su `addSpacing` asociado
      si queda hueco.
- [ ] T-007 · `marca`: `setFixedHeight(72)` y
      `AlignHCenter | AlignVCenter`.
- [ ] T-008 · `titulo` y `pie`: `setAlignment(AlignHCenter)`.
- [ ] T-009 · `pie` usa `ORGANIZACION` en lugar del literal
      «Ministerio de Educación Superior».

## Fase 3 · Tests

- [ ] T-010 · `test_theme.py`: helper `_contraste(hex_a, hex_b)` con la
      fórmula WCAG; aserciones de contraste de `blanco` sobre `lateral` (≥7)
      y de `sidebar_texto` sobre `lateral` (≥4.5).
- [ ] T-011 · `test_theme.py`: `escudo_pixmap()` por defecto mide 64 px de alto.
- [ ] T-012 · `test_theme.py`: `resources/icono-app.ico` existe, es un ICO
      válido y contiene los siete tamaños.
- [ ] T-013 · `test_navigation_ui.py`: el sidebar contiene exactamente una
      etiqueta con «Ministerio de Educación Superior — Cuba» y ninguna con
      «Ministerio de Educación Superior» a secas.
- [ ] T-014 · `test_navigation_ui.py`: el escudo está centrado
      (`x + ancho/2 ≈ sidebar.width()/2`, tolerancia 2 px).
- [ ] T-015 · `test_navigation_ui.py`: `test_el_sidebar_cabe_en_720` sigue
      pasando.

## Fase 4 · Verificación

- [ ] T-016 · `pytest -q -m "not slow"` en verde.
- [ ] T-017 · `grep -R "PySide6\|PyQt\|from ui\." core/` vacío.
- [ ] T-018 · `black --check` e `isort --check-only` en los ficheros tocados.
- [ ] T-019 · `python main.py` y comprobación visual del sidebar.
