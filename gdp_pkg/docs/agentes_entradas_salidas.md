# Entradas y Salidas por Agente — GDP Bloc 1

**Regla simple:** un **parámetro** es un dato conocido antes de resolver — no lo decide el modelo.
Una **variable** es una decisión que el optimizador elige.

---

## Agregador

| | Elemento | Tipo | Descripción |
|---|---|---|---|
| **Entradas** | `p_av`, `p_act`, `p_dev` | Parámetro | Precios GDP (dados por HQ) |
| | `η_max`, `η_min` | Parámetro | Límites contractuales |
| | `ω^s` | Parámetro | Probabilidades de escenario GDP |
| | `λ_k^n` | Entrada ADMM | Precio de clearing (viene del iterado anterior) |
| | `F_{i,k}^n` → `S_k` | Entrada ADMM | Oferta agregada FSPs (del paso paralelo) |
| **Variables** | `η` | Variable 1ª etapa | Potencia declarada al operador [kW] |
| | `C_max` | Variable 1ª etapa | Capacidad CLC contratada [kW] |
| | `D_k` | Variable 1ª etapa | Demanda de flexibilidad FSP por período [kWh] |
| | `c_k^s` | Variable 2ª etapa | Activación CLC por escenario [kWh] |
| | `r_k^s` | Variable 2ª etapa | Shortfall por escenario [kWh] |
| **Salidas** | `D_k^{n+1}`, `η`, `C_max` | Output ADMM | Enviados al loop para actualizar `λ_k` |

---

## FSP `i` (residencial)

| | Elemento | Tipo | Descripción |
|---|---|---|---|
| **Entradas** | `a_i`, `b_i`, `c_i` | Parámetro | Modelo térmico del edificio (privado) |
| | `α_i(t)` | Parámetro | Preferencia de incomodidad (privado) |
| | `u_i^{base}` | Parámetro | Consumo de referencia sin GDP |
| | `T_{out}^s` | Parámetro | Temperatura exterior por escenario |
| | `π_k · λ_k^n` | Entrada ADMM | Señal de precio (única información del mercado) |
| | `F_{i,k}^n` | Entrada ADMM | Propia oferta anterior (referencia proximal) |
| **Variables** | `F_{i,k}` | Variable 1ª etapa | Oferta de flexibilidad [kWh] — **lo único que comparte** |
| | `u_{i,t}` | Variable 1ª etapa | Potencia de calefacción [kW] |
| | `T_{i,t}^{in,s}` | Variable 2ª etapa | Temperatura interior por escenario [°C] |
| **Salidas** | `F_{i,k}^{n+1}` | Output ADMM | Enviada al agregador (sin revelar costos ni parámetros) |

---

## CLC Industrial (backup)

| | Elemento | Tipo | Descripción |
|---|---|---|---|
| **Entradas** | `C_max` | Parámetro | Capacidad máxima contratada por el agregador [kW] |
| | `p_CLC^l` | Parámetro | Costo de activación (LMP estocástico NY-HQ) |
| | `D_k`, `S_k` | Parámetro | Demanda del agregador y oferta FSP |
| **Variables** | `c_k^s` | Variable (del agregador) | Volumen de activación [kWh] — lo decide el agregador |
| **Salidas** | `q_k = D_k + c_k^s` | Output | Entrega total al operador de red |

---

## Flujo de información

```
HQ  ──[p_av, η_max, K_AM, K_PM]──▶  AGREGADOR
                                          │
                       λ_k (precio) ◀─────┤────▶ F_{i,k} (oferta)
                                          │           │
                                       FSP 1..N ──────┘
                                          │
                       gap (D_k > S_k) ───▶  CLC Industrial  ──▶  q_k total
```

> **Privacidad:** los FSPs solo exponen `F_{i,k}`.
> Sus parámetros térmicos `(a, b, c, α)` y función de costo nunca salen del agente.
