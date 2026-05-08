# Moving Horizon Estimation (MHE) Robusto con Pérdida de Huber

## Descripción

Este repositorio contiene la implementación de un **Moving Horizon Estimator robusto** que utiliza la **pérdida de Huber** para estimar estados de sistemas dinámicos en presencia de outliers. El trabajo incluye aplicaciones con datos de sensores GPS e IMU, comparando el desempeño de MHE clásico, MHE robusto y filtro de Kalman.

## Características Principales

- ✅ Implementación de MHE.
- ✅ Robustez ante outliers usando pérdida de Huber
- ✅ Integración con herramientas de optimización: **CasADi** y **FATROP**
- ✅ Sincronización automática de datos GPS e IMU
- ✅ Análisis comparativo de estimadores
- ✅ Simulaciones con perturbaciones y ruido realista

## Contenido del Repositorio

### Archivos Principales

| Archivo | Descripción |
|---------|-------------|
| `herramientas_opti.py` | Funciones para formular problemas MHE con CasADi |
| `GPS_MHE.ipynb` | Notebook: Aplicación de MHE robusto a datos GPS/IMU reales |
| `nonlinear_example.ipynb` | Notebook: Comparativa en sistema no lineal con 100 simulaciones |

### Datos

| Archivo | Descripción |
|---------|-------------|
| `gps_x_y_vx_vy.npz` | Datos de posición y velocidad del GPS |
| `imu_ax_ay.npz` | Datos de aceleración del IMU |

## Requisitos

```bash
Python >= 3.8
pip install casadi>=3.5.5
pip install numpy scipy matplotlib seaborn
pip install fatrop  # Opcional, para solver avanzado
```

## Uso Rápido

### 1. Ejecutar el ejemplo GPS con MHE robusto

```bash
jupyter notebook GPS_MHE.ipynb
```

### 2. Ejecutar análisis comparativo de estimadores

```bash
jupyter nonlinear_example.ipynb
```

## Metodología

El MHE resuelve en cada instante de tiempo un problema de optimización local:

$$\min_{\mathbf{x}, \mathbf{w}, \mathbf{v}} \left\| \mathbf{x}_0 - \hat{\mathbf{x}}_0 \right\|^2_P + \sum_{i=1}^{N_t} \left( L_\delta(\mathbf{v}_i) + \left\| \mathbf{w}_i \right\|_Q^2 \right)$$

donde $L_\delta$ es la **pérdida de Huber** para robustez ante outliers:

$$L_\delta(e) = \begin{cases} 
e^2 & |e| \leq \delta \\
2\delta|e| - \delta^2 & |e| > \delta
\end{cases}$$

## Características Especiales

### Robustez ante Outliers
La pérdida de Huber proporciona transiciones suaves entre errores cuadráticos (cercanos a 0) y penalizaciones lineales (lejanos), mejorando significativamente la estimación en presencia de outliers.

## Autores

Desarrollado como trabajo de investigación para congreso IEEE Latin American.

## Licencia

Este código se proporciona con fines de investigación y educación.
