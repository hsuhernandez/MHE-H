import casadi as cs
import numpy as np
import time
from scipy import linalg

def jacobiano(func, indep, dep=0, name=None):
    """Función que devuelve la función jacobiana de una función de Casadi, donde:
        func es la función de Casadi
        indep es el índice de la variable independiente
        dep es el índice de la variable dependiente (por defecto es 0)
        name es el nombre de la función jacobiana (por defecto es None)
       Y devuelve:
        la función jacobiana de la función de Casadi
    
    func should be a casadi.Function object. indep and dep should be the index
    of the independent and dependent variables respectively. They can be
    (zero-based) integer indices, or names of variables as strings.
    """
    if name is None:
        name = "jac_" + func.name()
    jacname = ["jac"]
    for (i, arglist) in [(dep, func.name_out), (indep, func.name_in)]:
        if isinstance(i, int):
            i = arglist()[i]
        jacname.append(i)
    jacname = ":".join(jacname)
    return func.factory(name, func.name_in(), [jacname])

def variables_y_parametros(opti, N, meas_handling=''):
    """ Función que define las variables y parámetros del MHE multirate, donde:
        opti: objeto casadi Opti
        N: dict con dimensiones del sistema
        meas_handling: '' -> están todas las mediciones presentes
                       'ignore' -> no tener en cuenta medición GPS ausente (no costo ni restricción)
                       'zero_holder' -> rellenar con última medición conocida dentro de la ventana
                       'use_model' -> rellenar valores GPS ausentes propagando x0 con f (w=0)
        Devuelve:
        X: variable de estados (Nx x N_t+1)
        W: variable de ruidos de proceso (Nw x N_t)
        V: variable de ruidos de medición (Nv x N_t+1)
        Y: parámetro de mediciones (Ny x N_t+1)
        U: parámetro de entradas (Nu x N_t)
        P: parámetro de matriz de peso de costo de arribo (Nx x Nx)
        X0: parámetro de estado inicial propuesto (Nx,)
        THETA: parámetro indicador booleano de si hay medición lenta en cada instante (N_t+1,)
               si meas_handling='ignore', sino None
    """
    # Defino dimensiones del sistema
    if 'w' not in N:
            N['w'] = N['x']
    N['v'] = N['y']
    # Variables:
    X = [] # estados
    W = [] # ruidos de proceso
    V = [] # ruidos de medición
    for i in range(N['t']):
        X.append(opti.variable(N['x']))
        V.append(opti.variable(N['v']))
        W.append(opti.variable(N['w']))

    X.append(opti.variable(N['x']))
    V.append(opti.variable(N['v']))

    X = cs.hcat(X)
    W = cs.hcat(W)
    V = cs.hcat(V)
    
    # Parámetros:
    Y = opti.parameter(N['y'], N['t']+1) # mediciones
    U = opti.parameter(N['u'], N['t']) # entradas
    P = opti.parameter(N['x'], N['x']) # matriz de peso de costo de arribo
    X0 = opti.parameter(N['x']) # estado inicial propuesto
    if meas_handling == 'ignore':
        THETA = opti.parameter(1, N['t']+1)
    else:
        THETA = None

    return X, W, V, Y, U, P, X0, THETA

def huber(a, rho):
    """Función que define la función de costo de Huber, donde:
        a es el vector de errores
        rho es el parámetro de Huber
       Y devuelve:
        la función de costo de Huber
    """
    norm_a = cs.fabs(a)
    quad = norm_a**2
    linear = 2*rho*norm_a - rho**2
    out = cs.if_else(norm_a <= rho, quad, linear)
    return cs.sum1(out)


def cuadratico(a, Q):
    """ Función que define la función de costo cuadrático, donde:
         a es el vector de errores
         Q es la matriz de peso
        Y devuelve:
         el costo cuadrático
    """
    return cs.mtimes([a.T, Q, a])

def costo_medicion(v, R_inv, theta=None, rho_huber=None, 
                            huber_idx=None, ignore_idx=None):
    """ Función que calcula el costo de medición considerando la ausencia de mediciones
        multirate, donde:
         v: vector de ruido de medición
         R_inv: inversa de la matriz de covarianza del ruido de medición
         theta: indicador binario de si hay medición lenta en este instante
         rho_huber: parámetro Huber, si es None no se usa Huber
         huber_idx: índices de las componentes a las que se aplica Huber,
                    si es None, se aplica a todas (si rho_huber no es None)
         ignore_idx: índices de las mediciones que se deben ignorar
        Y devuelve:
         el costo de medición para un instante determinado
    """
    # Cantidad de mediciones
    n_v = v.size()[0]
    all_idx = np.arange(n_v)  # Todos los índices de mediciones

    # El resto del costo es cuadrático estándar (para las mediciones no indicadas)
    quad_idx = np.setdiff1d(all_idx, huber_idx)  # Indices que no tienen Huber
    if theta is None:  # Todas las mediciones presentes
        # Aplicamos Huber si es necesario
        h_cost = 0 if rho_huber is None else huber(v[huber_idx], rho_huber)
        # Costo cuadrático para el resto de las mediciones
        q_cost = 0 if len(quad_idx)==0 else cuadratico(v[quad_idx], R_inv[np.ix_(quad_idx, quad_idx)])
    else: # No tener en cuenta medición ausente (meas_handling='ignore')
        
        # Índices de mediciones a que no se ignoran
        rest_idx = np.setdiff1d(all_idx, ignore_idx)
        
        # Costo de Huber
        hr_idx = np.intersect1d(huber_idx, rest_idx)  # Índices de Huber que no se ignoran
        h_cost = cs.if_else(theta==1,
                    # Si theta==1, se usa todo el costo de Huber
                    0 if rho_huber is None else huber(v[huber_idx], rho_huber),
                    # Si theta==0, costo de Huber solo en las componentes no ignoradas
                    0 if len(hr_idx)==0 else huber(v[hr_idx], rho_huber))
        
        # Costo cuadrático
        qr_idx = np.intersect1d(quad_idx, rest_idx)  # Índices de costo cuadrático que no se ignoran
        q_cost = cs.if_else(theta==1,
                    # si theta==1, se usa todo el costo cuadrático
                    0 if len(quad_idx)==0 else cuadratico(v[quad_idx], R_inv[np.ix_(quad_idx, quad_idx)]),
                    # si theta==0, costo cuadrático solo en las componentes no ignoradas
                    0 if len(qr_idx)==0 else cuadratico(v[qr_idx], R_inv[np.ix_(qr_idx, qr_idx)]), 
                    )
    return h_cost + q_cost
    
def costo_y_restricciones(opti, N, f, h, X, W, V, Y, U, P, X0,
                          THETA=None, Q_inv=None, R_inv=None,
                          rho_huber=None, huber_idx=None, ignore_idx=None,
                          w_pos=False):
    """ Función que define el costo y las restricciones del MHE multirate, donde:
         opti: objeto casadi Opti
         N: dict con dimensiones del sistema
         f: función de proceso
         h: función de medición
         X: variable de estados (Nx x N_t+1)
         W: variable de ruidos de proceso (Nw x N_t)
         V: variable de ruidos de medición (Nv x N_t+1)
         Y: parámetro de mediciones (Ny x N_t+1)
         U: parámetro de entradas (Nu x N_t)
         P: parámetro de matriz de peso de costo de arribo (Nx x Nx)
         X0: parámetro de estado inicial propuesto (Nx,)
         THETA: parámetro indicador booleano de si hay medición lenta en cada instante (N_t+1,)
                si meas_handling='ignore', sino None
         Q_inv: inversa de la matriz de covarianza del ruido del proceso
         R_inv: inversa de la matriz de covarianza del ruido de medición
         rho_huber: parámetro Huber, si es None no se usa Huber
         huber_idx: índices de las componentes a las que se aplica Huber, si es None,
                    se aplica a todas (si rho_huber no es None)
         ignore_idx: índices de las mediciones que se deben ignorar, si es None, 
                     se ignoran todas (si theta no es None)
         w_pos: es la desviación estándar del ruido de proceso (si el ruido es positivo)
        Y devuelve:
            J: expresión del costo total"""
    # Defino dimensiones del sistema
    if 'w' not in N:
            N['w'] = N['x']
    N['v'] = N['y']

    # Si rho_huber es distinto de None y si huber_idx es None, 
    # entonces se aplica Huber a todas las mediciones
    if rho_huber is not None and huber_idx is None:
        huber_idx = np.arange(N['y'])
    
    if THETA is not None:
        # Si THETA es distinto de None y si ignore_idx es None,
        # entonces se ignoran todas las mediciones (cuando THETA es 0)
        if ignore_idx is None:
            ignore_idx = np.arange(N['y'])
        # índices de mediciones que no se ignoran
        rest_idx = np.setdiff1d(np.arange(N['y']), ignore_idx)

    # Costo y restricciones:
    J = cuadratico(X[:, 0]-X0, P)  # costo de arribo

    mu_w = float(w_pos)*np.sqrt(2 / np.pi) if w_pos else 0 # media del ruido de proceso si es no negativo, sino 0

    for k in range(N['t']):
        W_i = W[:, k] - mu_w if w_pos else W[:, k]
        J += cuadratico(W_i, Q_inv) # costo de ruido de proceso

        J += costo_medicion(V[:, k], R_inv, THETA[k] if THETA is not None else None, 
                            rho_huber=rho_huber, huber_idx=huber_idx, ignore_idx=ignore_idx)
        # restricciones de dinamica (siempre)
        opti.subject_to(X[:, k+1] == f(X[:, k], U[:, k], W[:, k]))
        if THETA is None: # Pongo todas las mediciones
            opti.subject_to(Y[:, k] == h(X[:, k]) + V[:, k])
        else:
            # Mediciones que no se ignoran (rápidas):
            if len(rest_idx) > 0:
                opti.subject_to(Y[rest_idx, k] == h(X[:, k])[rest_idx] + V[rest_idx, k])
            # Mediciones que pueden ignorarse (lentas):
            constraint = (Y[ignore_idx, k] - (h(X[:, k])[ignore_idx] + V[ignore_idx, k]))
            opti.subject_to(THETA[k] * constraint == cs.DM.zeros(len(ignore_idx)))
    
    # último instante
    J += costo_medicion(V[:, N['t']], R_inv, THETA[N['t']] if THETA is not None else None, 
                        rho_huber=rho_huber, huber_idx=huber_idx, ignore_idx=ignore_idx)
    if THETA is None: # Pongo todas las mediciones
        opti.subject_to(Y[:, N['t']] == h(X[:, N['t']]) + V[:, N['t']])
    else:
        # Mediciones que no se ignoran (rápidas):
        if len(rest_idx) > 0:
            opti.subject_to(Y[rest_idx, N['t']] == h(X[:, N['t']])[rest_idx] + V[rest_idx, N['t']])
        # Mediciones que pueden ignorarse (lentas):
        constraint = (Y[ignore_idx, N['t']] - (h(X[:, N['t']])[ignore_idx] + V[ignore_idx, N['t']]))
        opti.subject_to(THETA[N['t']] * constraint == cs.DM.zeros(len(ignore_idx)))

    # opti.subject_to(opti.bounded(-cs.inf, X, cs.inf))
    # opti.subject_to(opti.bounded(-cs.inf, W, cs.inf))
    # opti.subject_to(opti.bounded(-cs.inf, V, cs.inf))
    return J

def ekf(f, h, x, u, w, y, P, Q, R, f_jacx=None, f_jacw=None, h_jacx=None):
    """Función que implementa el filtro de Kalman extendido, donde:
        f es la función de proceso
        h es la función de medición
        x es el estado, xhat(k-1 | k-1)
        u es la entrada, u(k-1)
        w es el ruido de proceso, w(k-1)
        y es la medición, y(k)
        P es la matriz de covarianza del estado P(k-1 | k-1)
        Q es la matriz de covarianza del ruido de proceso
        R es la matriz de covarianza del ruido de medición
        f_jacx es la jacobiana de f respecto a x
        f_jacw es la jacobiana de f respecto a w
        h_jacx es la jacobiana de h respecto a x
        (si no se especifican, se calculan automáticamente)
       Y devuelve:
        x_upd el estado actualizado, xhat(k | k)
        P_upd la matriz de covarianza del estado actualizado, P(k | k)
    """
    # Defino jacobianos
    if f_jacx is None:
        f_jacx = jacobiano(f, 0)
    if f_jacw is None:
        f_jacw = jacobiano(f, 2)
    if h_jacx is None:
        h_jacx = jacobiano(h, 0)
    
    # Predicción
    x_pred = f(x, u, w) # estado predicho, xhat(k | k-1) 
    F = f_jacx(x, u, w) # jacobiana de f respecto a xkat(k-1 | k-1)
    G = np.array(f_jacw(x, u, w)) # jacobiana de f respecto a w (k-1 | k-1)
    P_pred = cs.mtimes([F,P,F.T]) + cs.mtimes([G,Q,G.T]) # Covarianza de estimación predicha, P(k | k-1)
    
    # Actualización
    y_tilde = y - h(x_pred) # residuo de la medición
    H = h_jacx(x_pred) # jacobiana de h respecto a xhat(k | k-1)
    S = cs.mtimes([H,P_pred,H.T])+R # covarianza de innovación (o residual)
    K = cs.mtimes([P_pred,H.T,linalg.inv(S)]) # ganancia de Kalman


    # forma tradicional
    x_upd = x_pred + cs.mtimes(K, y_tilde) # estado actualizado, xhat(k | k)
    P_upd = cs.mtimes((cs.DM.eye(P_pred.shape[0])-cs.mtimes(K,H)), P_pred) # Covarianza de estimación actualizada, P(k | k)
    
    return x_upd, P_upd

def adaptative_gain(update_P, x0, P, y, h, **kwargs):
    xPx = cs.mtimes([x0.T, P, x0])
    PxxP = cs.mtimes([P,x0, x0.T, P])

    if update_P == 'AD-CF': # Adaptative gain, constant forgetting factor
        alpha = kwargs.get('alpha', 0.95)
        beta = kwargs.get('beta', 1)
        P = 1/alpha * (P - (PxxP)/(alpha/beta + xPx))

    elif update_P == 'AD-CT': # Adaptative gain, constant trace
        Xi = kwargs.get('Xi', 20)
        eta = kwargs.get('eta', 3)  
        alpha = 1/Xi * np.trace(P-xPx/(eta+xPx))
        beta = eta**-1 * alpha
        den = alpha/beta + xPx
        if np.abs(den) < 1e-8 or alpha < 1e-8:
            P = P + 1e-6 * cs.DM.eye(P.shape[0])  # fallback suave
        else:
            P = 1/alpha * (P - PxxP / den)
        return P
    
    elif update_P == 'AD-VF': # Adaptative gain, variable forgetting factor
        sigma = kwargs.get('sigma', 100)
        c = kwargs.get('c', 10e6)
        meas_err = np.linalg.norm(y - h(x0), ord=2)**2 # + eps
        Nag = (1.0 + xPx) * (sigma / meas_err)
        alpha = 1.0 - 1.0/Nag
        W = P - PxxP/(1.0 + xPx)
        if 1/alpha*np.trace(W) <= c:
            P = 1/alpha * W
        else:
            P = W
    
    return P

def actualizar_x0_P(update_P, x0, P0, x1, u, w, y, f, h, Q, R, 
                    f_jacx=None, f_jacw=None, h_jacx=None, **kwargs):
    if not update_P:
        return x1, P0
    elif update_P == 'EKF':
        _, P = ekf(f, h, x0, u, w, y, P0, Q, R, f_jacx, f_jacw, h_jacx)
        return x1, P
    else:
        P = adaptative_gain(update_P, x0, P0, y, h, **kwargs)
        return x1, P
    

def mhe(N, f, h, x0, u, y, P0, Q, R, theta_buffer = None,
        f_jacx=None,f_jacw=None,h_jacx=None, update_P=False, 
        rho_huber = None, huber_idx = None, meas_handling='', 
    ignore_idx=None, solver = 'fatrop', w_pos=False, **kwargs):
    """Función que implementa el estimador de horizonte móvil multirate, donde:
    N: dict con dimensiones del sistema
    f, h: casadi functions
    x0: estado inicial (numpy)
    u: entradas (Nu x Nsim)
    y: mediciones (Ny x Nsim) con NaN para mediciones ausentes
    P0: matriz de peso de costo de arribo (numpy)
    Q, R: matrices de covarianza de ruidos de proceso y medición (numpy)
    theta_buffer: int array (Nsim,) indicando con 1 o 0 en cada instante hay medición lenta
                    si theta_buffer es None, se asume que todas las mediciones están presentes
    f_jacx, f_jacw, h_jacx: jacobianas de f y h (casadi functions), si no se especifican,
                            se calculan automáticamente
    update_P: Método para actualizar adaptativamente P.
    rho_huber: parámetro Huber, si es None no se usa Huber
    huber_idx: índices de las componentes a las que se aplica Huber, si es None,
               se aplica a todas (si rho_huber no es None)
    meas_handling: '' -> están todas las mediciones presentes
                   'ignore' -> no tener en cuenta medición ausente (no costo ni restricción)
                   'zero_holder' -> rellenar con última medición conocida dentro de la ventana
                   'use_model' -> rellenar valores ausentes propagando x0 con f (w=estimado)
    ignore_idx: índices de las mediciones que se deben ignorar, si es None, 
                se ignoran todas (si theta no es None)
    solver: solver de casadi a usar ('ipopt' o 'fatrop')
    w_pos: si True, impone ruido de proceso no negativo en todas las componentes
    **kwargs: parámetros adicionales para la actualización adaptativa de P
    Devuelve:
    x_mhe: estados estimados (Nx x Nsim)
    w_mhe: ruidos de proceso estimados (Nw x Nsim-1)
    v_mhe: ruidos de medición estimados (Nv x Nsim)
    t_mhe: tiempos de cómputo por iteración (Nsim-N['t'],)
    y_used: mediciones usadas en cada iteración (Ny x Nsim)
    """
    # Defino dimensiones del sistema
    if 'w' not in N:
            N['w'] = N['x']
    N['v'] = N['y']

    Nsim = y.shape[1]  # cantidad de mediciones

    opti = cs.Opti()

    # Variables y parámetros:
    X, W, V, Y, U, P, X0, THETA = variables_y_parametros(opti, N, meas_handling=meas_handling)
    
    Q_inv = linalg.inv(Q) # inversa de Q
    R_inv = linalg.inv(R) # inversa de R
    # Costo y restricciones:
    J = costo_y_restricciones(opti, N, f, h, X, W, V, Y, U, P, X0, THETA, Q_inv, R_inv, 
                                        rho_huber=rho_huber, huber_idx=huber_idx,
                                        ignore_idx=ignore_idx, w_pos=w_pos)

    if solver == 'ipopt':
        opts_setting = {'ipopt.max_iter':2000, 'ipopt.print_level':0, 'print_time':0, 
                        'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
        opti.solver('ipopt', opts_setting)
    elif solver == 'fatrop':
        options = {}
        options["print_time"] = False
        options["expand"] = True
        options["print_out"] = False
        options["fatrop"] = {"mu_init": 0.1, "print_level": 0}
        options["structure_detection"] = "auto"
        options["debug"] = True
        options["verbose"] = 0
        opti.solver("fatrop", options)
    opti.minimize(J)

    # inicializo primer ventana:
    t_ini = time.time()
    y_window = y[:, 0:N['t']+1].copy()

    if meas_handling != '': # relleno mediciones ausentes
        y_window = _iniciar_ventana(y_window, x0, u_window=u[:, 0:N['t']], 
                                    f=f, h=h, meas_handling=meas_handling)

    opti.set_value(Y, y_window)
    opti.set_value(P, P0)
    opti.set_value(U, u[:, 0:N['t']])
    opti.set_value(X0, x0)
    if THETA is not None:
        opti.set_value(THETA, theta_buffer[0:N['t']+1])

    sol = opti.solve()
    x_first = sol.value(X)
    v_first = sol.value(V)
    w_first = sol.value(W)

    n_out = Nsim - N['t']
    x_mhe = np.empty((N['x'], Nsim))
    v_mhe = np.empty((N['v'], Nsim))
    w_mhe = np.empty((N['w'], Nsim-1))
    t_mhe = np.empty(n_out)
    y_used = np.empty((N['y'], Nsim))

    # Inicialización con la primera ventana resuelta
    x_mhe[:, :N['t']+1] = x_first
    v_mhe[:, :N['t']+1] = v_first
    w_mhe[:, :N['t']] = w_first
    y_used[:, :N['t']+1] = y_window
    t_mhe[0] = time.time() - t_ini

    for i in range(1, Nsim-N['t']):
        t_ini = time.time()
        x0, P0 = actualizar_x0_P(update_P, sol.value(X[:, 0]).reshape( N['x'], 1), P0, sol.value(X[:, 1]), 
                                u[:, i-1], sol.value(W[:, 0]), y_window[:, 0],
                                f, h, Q, R, f_jacx, f_jacw, h_jacx, **kwargs)
        y_window = y_window[:, 1:]  # desplazo ventana
        y_window = np.column_stack((y_window, y[:, i+N['t']]))
        if meas_handling != '': # relleno mediciones ausentes
            y_window = _actualizar_ventana(y_window, x_mhe[:, i+N['t']-1], 
                                           u_kminus1 = u[:, i+N['t']-2], 
                                           w_kminus1 = w_mhe[:, i+N['t']-2],
                                           f=f, h=h, meas_handling=meas_handling)
        opti.set_value(Y, y_window)
        opti.set_value(P, P0)
        opti.set_value(U, u[:, i:i+N['t']])
        opti.set_value(X0, x0)
        if THETA is not None:
            opti.set_value(THETA, theta_buffer[i:i+N['t']+1])

        opti.set_initial(X, sol.value(X))
        opti.set_initial(W, sol.value(W))
        opti.set_initial(V, sol.value(V))

        sol = opti.solve()
    
        x_mhe[:, i+N['t']] = sol.value(X[:, N['t']])
        v_mhe[:, i+N['t']] = sol.value(V[:, N['t']])
        w_mhe[:, i+N['t']-1] = sol.value(W[:, N['t']-1])
        t_mhe[i] = time.time() - t_ini
        y_used[:, i+N['t']] = y_window[:, -1]
    return x_mhe, w_mhe, v_mhe, t_mhe, y_used

def _iniciar_ventana(y_window, x_0=None, u_window=None, f=None, h=None, meas_handling='ignore'):
    """Solución para iniciar la ventana de mediciones en MHE multirate,
    rellenando las mediciones ausentes.
    Donde:
        y_window: matriz de mediciones en la ventana (Ny x N_t+1)
        x_0: estado inicial (Nx,)
        u_window: matriz de entradas en la ventana (Nu x N_t)
        f: función de estado
        h: función de medición
        meas_handling: 'ignore' -> no tener en cuenta medición ausente (no costo ni restricción)
                       'zero_holder' -> rellenar con última medición conocida dentro de la ventana
                       'use_model' -> rellenar valores ausentes propagando x0 con f (w=0)
    Devuelve:
        y_out: matriz de mediciones rellenadas (Ny x N_t+1)
    """
    y_out = y_window.copy()
    if meas_handling == 'ignore':
        y_out[np.isnan(y_out)] = 0
        return y_out
    
    elif meas_handling == 'zero_holder':
        for k in range(1, y_out.shape[1]):
            y_out[:, k] = np.where(np.isnan(y_out[:, k]), y_out[:, k-1], y_out[:, k])
        # si las primeras mediciones son nan, rellenar con h(f(x_k, u_k, 0))
        if any(np.isnan(y_out[:, 0])):
            x_k = x_0
            k = 0
            w_zero = np.zeros((x_k.shape))  # ruido de proceso cero
            while(any(np.isnan(y_out[:, k]))) and k < y_out.shape[1]-1:
                y_k = h(x_k).full().flatten()
                y_out[:, k] = np.where(np.isnan(y_out[:, k]), y_k, y_out[:, k])
                x_k = f(x_k, u_window[:, k], w_zero)
                k += 1
        return y_out
    
    elif meas_handling == 'use_model':
        # Creo un arreglo y_tilde para las mediciones rellenadas
        x_k = x_0
        w_zero = np.zeros((x_k.shape))  # ruido de proceso ceroo
        for k in range(y_out.shape[1]-1):
            y_k = h(x_k).full().flatten()
            y_out[:, k] = np.where(np.isnan(y_out[:, k]), y_k, y_out[:, k])
            x_k = f(x_k, u_window[:, k], w_zero)
        # Última medición
        y_k = h(x_k).full().flatten()
        y_out[:, -1] = np.where(np.isnan(y_out[:, -1]), y_k, y_out[:, -1])
        return y_out

def _actualizar_ventana(y_window, x_kminus1=None, u_kminus1=None, w_kminus1=None,
                        f=None, h=None, meas_handling='ignore'):
    """Solución para actualizar la ventana de mediciones en MHE multirate,
    rellenando la última medición.
    Donde:
        y_window: matriz de mediciones en la ventana (Ny x N_t+1)
        x_kminus1: estado en el último instante de la ventana anterior (Nx,)
        u_kminus1: entrada en el último instante de la ventana anterior (Nu,)
        w_kminus1: ruido de proceso estimado en el último instante de la ventana anterior (Nw,)
        f: función de estado
        h: función de medición
        meas_handling: 'ignore' -> no tener en cuenta medición GPS ausente (no costo ni restricción)
                       'zero_holder' -> rellenar con última medición conocida dentro de la ventana
                       'use_model' -> rellenar valores GPS ausentes propagando x0 con f (w=estimado)
    Devuelve:
        y_out: matriz de mediciones rellenadas (Ny x N_t+1)
    """
    y_out = y_window.copy()
    if meas_handling == 'ignore':
        y_out[:, -1] = np.where(np.isnan(y_out[:, -1]), 0, y_out[:, -1])
        return y_out
    
    elif meas_handling == 'zero_holder':
        if np.any(np.isnan(y_out[:, -1])):
            y_out[:, -1] = np.where(np.isnan(y_out[:, -1]), y_out[:, -2], y_out[:, -1])
        return y_out
    
    elif meas_handling == 'use_model':
        if np.any(np.isnan(y_out[:, -1])):
            w_zero = np.zeros((w_kminus1.shape))
            x_k = f(x_kminus1, u_kminus1, w_zero)
            y_k = h(x_k).full().flatten()
            y_out[:, -1] = np.where(np.isnan(y_out[:, -1]), y_k, y_out[:, -1])
        return y_out
    

def resolver_ekf(N, f, h, x_0, u, y, P, Q, R, 
                 f_jacx=None, f_jacw=None, h_jacx=None, 
                 w_pos=False, theta_buffer=None,
                 meas_handling=''):
    """"""
    # Defino dimensiones del sistema
    if 'w' not in N:
            N['w'] = N['x']
    
    # Defino jacobianos
    if f_jacx is None:
        f_jacx = jacobiano(f, 0)
    if f_jacw is None:
        f_jacw = jacobiano(f, 2)
    if h_jacx is None:
        h_jacx = jacobiano(h, 0)

    Nsim = len(np.array(y)[0,:]) # Cantidad de mediciones
    x_ekf = cs.DM.zeros(N['x'], Nsim) # estimados de x
    x_ekf[:,0] = x_0 # estado inicial
    P_0 = P
    w_0 = cs.DM.zeros(N['w'], 1) # ruido de proceso inicial
    t_ekf = np.array([]) # tiempos de cómputo por iteración
    
    if w_pos:
        mu_w = w_pos*np.sqrt(2/np.pi)
        w_0 += mu_w*cs.DM.ones(N['w'],1) # ruido de proceso inicial
    
    y_used = y.copy()

    if theta_buffer is None:
        theta_buffer = np.ones(Nsim)
    for i in range(Nsim-1):
        t_ini = time.time()
        if theta_buffer[i] == 0: # medición lenta ausente
            if meas_handling == 'zero_holder':
                y_used[:, i] = y_used[:, i-1]
            elif meas_handling == 'use_model':
                x_pred = f(x_ekf[:, i-1], u[:, i], w_0)
                y_used[:, i] = h(x_pred).full().flatten()
            
        x, P_0 = ekf(f,h, x_ekf[:, i], u[:, i], w_0,  y_used[:,i],
                     P_0, Q, R, f_jacx=f_jacx, f_jacw=f_jacw, h_jacx=h_jacx)
        x_ekf[:,i+1] = x
        t_ekf = np.concatenate((t_ekf, [time.time() - t_ini]))
    
    return np.array(x_ekf), t_ekf, y_used
