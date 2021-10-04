#!/usr/bin/env python
# coding: utf-8

# # Gradient Descent

# ## Überblick

# Wir betrachten das Gradient-Descent Verfahren 
# \begin{equation*}
# x_{t+1} = x_t - \gamma_t f'_t,
# \end{equation*}
# untersuchen Konvergenz, d.h.
# \begin{equation*} 
# f_t - f_\ast \xrightarrow{t\to\infty} 0
# \end{equation*}
# bzw.
# \begin{equation*} 
# \|x_t - x_\ast\| \xrightarrow{t\to\infty} 0
# \end{equation*}  
# und versuchen das asymptotische Verhalten genauer zu analysieren.
#   
# Für den Rest des Kapitels setzen wir $f\in C^1(\mathbb{R}^d)$ konvex und $\gamma_t = \gamma$ konstant voraus.

# ## Vorüberlegungen

# Ist
#   $f\in C^1(\mathbb{R}^d)$, 
#   $x_\ast = \mathrm{argmin}_{x\in\mathbb{R}}f(x)$, 
#   $x_{t+1} = x_t - \gamma f'_t$, dann gilt
#   \begin{equation*} 
#   \begin{aligned}
#   \|x_{t+1}-x_{*}\|_{2}^{2} 
#   &=\|x_{t}-x_{*}-\gamma f_{t}^{\prime}\|_{2}^{2} \\
#   &=\|x_{t}-x_{*}\|_{2}^{2}+\gamma^{2}\|f_{t}^{\prime}\|_{2}^{2}-2
#   \gamma f_{t}^{\prime}(x_{t}-x_{*})
#   \end{aligned}
#   \end{equation*}
#   und somit
#   \begin{equation*} 
#   f'_{t}(x_{t}-x_{*}) 
#   =\frac{1}{2 \gamma}\big(\gamma^{2}\|f_{t}^{\prime}\|_{2}^{2}
#   +\|x_{t}-x_{*}\|_{2}^{2}-\|x_{t+1}-x_{*}\|_{2}^{2}\big).
#   \end{equation*}
# 
# Für konvexes $f$  ist
#   \begin{equation*} 
#   f(y) \geq f(x) + f'(x)(y-x)
#   \end{equation*}
#   und mit $y=x_\ast$, $x=x_t$
#   \begin{equation*} 
#   f_\ast \geq f_t + f'_t (x_\ast-x_t)
#   \end{equation*}
#   bzw.
#   \begin{align*} 
#   0 
#   \leq f_t - f_\ast
#   &\leq f'_t(x_t - x_\ast)\\
#   &= \frac{1}{2 \gamma}\big(\gamma^{2}\|f_{t}^{\prime}\|_{2}^{2}
# +\|x_{t}-x_{*}\|_{2}^{2}-\|x_{t+1}-x_{*}\|_{2}^{2}\big).
#   \end{align*}

# Aufsummiert erhält man
# \begin{align*} 
# \sum_{t=0}^{T-1} (f_t - f_\ast)
# & \leq \sum_{t=0}^{T-1} f'_t(x_t - x_\ast)\\
# &   =   \frac{1}{2 \gamma}
#         \ \sum_{t=0}^{T-1} 
#          \big( \gamma^{2}\|f_{t}^{\prime}\|_{2}^{2}
#                 +\|x_{0}-x_\ast\|_{2}^{2}
#                 -\|x_{t+1}-x_{*}\|_{2}^{2}
#           \big)
# \end{align*}
# bzw.
# \begin{align*} 
# \underbrace{\frac{1}{T}\sum_{t=0}^{T-1} (f_t - f_\ast)}_{\text{''mittlere Abweichung von } f_\ast  \text{''}}
# & \leq 
#   \frac{\gamma}{2}
#   \underbrace{\frac{\sum_{t=0}^{T-1} \|f_{t}^{\prime}\|_{2}^{2}}{T}}_\text{''mittlerer Gradient''} 
# \\
# & \quad +
#   \frac{1}{2 \gamma}
#   \underbrace{\frac{\|x_{0}-x_\ast\|_{2}^{2}-\left\|x_{T}-x_{*}\right\|_{2}^{2}}{T}}_{\leq \frac{\|x_{0}-x_\ast\|_{2}^{2}}{T} = \mathcal{O}(\frac{1}{T})\ \text{''Startfehler''}}.
# \end{align*}

# Mit $\hat{t} = \mathrm{argmin}_{t\in\{0,\ldots,T-1\}}(f_t - f_\ast)$ ($\hat{t}$ nicht notwendig gleich $T-1$) folgt
#   \begin{equation*} 
#   f_{\hat{t}} - f_\ast \leq \frac{1}{T}\sum_{t=0}^{T-1} (f_t - f_\ast).
#   \end{equation*}
# Der Anteil 
#   \begin{equation*} 
#   \frac{1}{2 \gamma}\frac{\|x_{0}-x_\ast\|_{2}^{2}-\|x_{T}-x_{*}\|_{2}^{2}}{T}
#   \end{equation*} 
#   war zu erwarten.
# Das Ziel ist es nun
#   \begin{equation*} 
#   \frac{\gamma}{2T} \sum_{t=0}^{T-1} \|f_{t}^{\prime}\|_{2}^{2}
#   \end{equation*}
#   zu kontrollieren.
# Dazu muss $f$ zusätzliche Voraussetzungen erfüllen.

# ## Lipschitz-Stetigkeit

# Im letzten Abschnitt haben wir  $f\in C^1(\mathbb{R}^d)$ konvex vorausgesetzt.
# Jetzt fordern wir zusätzlich Lipschitz-Stetigkeit von $f$, d.h.
# \begin{equation*} 
# |f(y) - f(x)| \leq L_f \|y - x\| \quad \forall x,y\in\mathbb{R}^d.
# \end{equation*}
# Dies ist äquivalent zur Beschränkheit des Gradienten, wie das folgende
# Ergebnis aus der Analysis zeigt.

# **Lemma:** $f:\mathbb{R}^d \supset \mathrm{dom}(f) \rightarrow \mathbb{R}$ differenzierbar
# (nicht notwendig konvex), $X\subset\mathrm{dom}(f)$ offen, konvex.
# Dann ist
# \begin{equation*} 
# |f(x) -f(y)| \leq L_f \|x-y\| \quad \forall x,y\in X
# \end{equation*}
# äquivalent zu
# \begin{equation*} 
# \|f'(x)\| \leq L_f \quad \forall x\in X,
# \end{equation*}
# wobei bei $f'$ die induzierte Operatornorm benutzt wird.

# **Beweis:**
# 
# "$\Rightarrow$"
# 
# - für $f$ gelte
#     \begin{equation*} 
#     |f(x) -f(y)| \leq L_f \|x-y\| \quad \forall x,y\in X
#     \end{equation*}
# 
# - da $X$ offen ist gibt es für jedes $x\in X$ eine Kugel
# $B_r(x)$ mit ${B}_r(x)\subset X$
# 
# - für beliebiges $v\in\mathbb{R}^d$ mit $\|v\|=1$ ist deshalb die Funktion
#     \begin{equation*} 
#     g(t) = f(x + tv), \quad t \in (-r,r)
#     \end{equation*}
#   wohldefiniert
# 
# - mit $f$ ist auch $g$ differenzierbar mit
#     \begin{equation*} 
#     g'(t) = f'(x + tv)v
#     \end{equation*}
#   und somit gilt für alle $v\in\mathbb{R}^d$ mit $\|v\|=1$
#     \begin{align*}
#     \|f'(x)v\|
#     &= |g'(0)| \\
#     &= \big| \lim_{t\to 0} \frac{g(t) - g(0)}{t}  \big| \\
#     &= \lim_{t\to 0}  \big| \frac{f(x + tv) - f(x)}{t}  \big| \\
#     &\leq L_f \lim_{t\to 0}  \big\| \frac{x + tv - x}{t}  \big\| \\
#     & = L_f \|v\|,
#     \end{align*}
#   also
#     \begin{equation*} 
#     \|f'(x)\| \leq L_f 
#     \end{equation*}
# 
# "$\Leftarrow$"
# 
# - für $f$ gelte
#     \begin{equation*} 
#     \|f'(x)\| \leq L_f \quad \forall x\in X
#     \end{equation*}
# 
# - da $X$ konvex ist, ist für alle $x,y\in X$ und $t\in[0,1]$ die Funktion
#     \begin{equation*} 
#     g(t) = f\big(x+ t(y-x)\big)
#     \end{equation*}
#   wohldefiniert und es gilt
#     \begin{equation*} 
#     g(0) = f(x),
#     \quad
#     g(1) = f(y)
#     \end{equation*}
# 
# - mit $f$ ist auch $g$ differenzierbar mit
#     \begin{equation*} 
#     g'(t) = f'\big(x+ t(y-x)\big)(y-x)
#     \end{equation*}
# 
# - durch Anwendung des Mittelwertsatzes folgt
#     \begin{align*}
#     |f(x) - f(y)| 
#     & = |g(1) - g(0)| \\
#     & = |g'(\tau)|\\
#     & = |f'\big(\underbrace{x+ \tau (y-x)}_{\xi}\big)(y-x)| \\
#     & \leq \|f'(\xi)\| \ \|y-x\| \\
#     & \leq L_f \|x-y\| 
#     \end{align*}
# 
# $\square$

# Setzen wir dies in die summierte Abschätzung von oben ein, so erhalten wir
#   \begin{align*}
#   \sum_{t=0}^{T-1} (f_t - f_\ast)
#   &\leq \sum_{t=0}^{T-1}f'_t(x_t - x_\ast)\\
#   & = \frac{1}{2 \gamma}
#       \ \sum_{t=0}^{T-1} 
#    \big(\gamma^{2}\|f_{t}^{\prime}\|_{2}^{2}+\|x_{0}-x_\ast\|_{2}^{2}-\|x_{t+1}-x_{*}\|_{2}^{2}\big)\\
#   & \leq \frac{\gamma}{2}T L_f^2 
#   + \frac{1}{2\gamma} 
#   \big( 
#   \underbrace{\|x_{0}-x_{*}\|_{2}^{2}}_{e_0^2}
#   -
#   \underbrace{\|x_{T}-x_{*}\|_{2}^{2}}_{\geq 0}
#   \big)\\
#   &\leq \frac{\gamma T L_f^2}{2} + \frac{e_0^2}{2\gamma},\\
#   \end{align*}
# also
#   \begin{align*} 
#   \min_{t=0,\ldots,T-1}(f_t - f_\ast)
#   \leq \frac{1}{T} \sum_{t=0}^{T-1} (f_t - f_\ast)
#   \leq \frac{\gamma L_f^2}{2} + \frac{e_0^2}{2\gamma T}.
#   \end{align*}

# Wann verschwindet die rechte Seite für $T\to\infty$ ?
# Beide Summanden auf der rechten Seite sind $\geq 0$, so dass
#   \begin{equation*} 
#   \frac{\gamma L_f^2}{2}\xrightarrow{T\to\infty}0,
#   \quad
#   \frac{e_0^2}{2\gamma T}\xrightarrow{T\to\infty}0
#   \end{equation*}
# gelten muss, also
#   \begin{equation*} 
#   \gamma \xrightarrow{T\to\infty}0, \quad \gamma T \xrightarrow{T\to\infty}\infty.
#   \end{equation*}
# Mit dem Ansatz
#   \begin{equation*} 
#   \gamma = \frac{c}{T^\omega}, \quad c,\omega > 0
#   \end{equation*}
# gilt immer $\gamma \xrightarrow{T\to\infty}0$.

# Für den zweiten Teil erhalten wir 
#   $\gamma T = c T^{1-\omega}\xrightarrow{T\to\infty}\infty$
# falls  $1-\omega>0$, also 
#   \begin{equation*} 
#   \omega < 1
#   \end{equation*}
# ist.
# Oben eingesetzt folgt
#   \begin{align*} 
#   \min_{t=0,\ldots,T-1}(f_t - f_\ast)
#   &\leq \frac{\gamma L_f^2}{2} + \frac{e_0^2}{2\gamma T} \\ 
#   & = 
#   \frac{c L_f^2}{2}\frac{1}{T^\omega} + \frac{e_0^2}{2c} \frac{1}{T^{1-\omega}} \\
#   &= \mathcal{O}\Big(\big(\frac{1}{T}\big)^{\min(\omega,1-\omega)}\Big).
#   \end{align*}

# Die obere Schranke
#   \begin{equation*} 
#   g(\gamma) = \frac{\gamma L_f^2}{2} + \frac{e_0^2}{2\gamma T}
#   \end{equation*}
# wird wegen
#   \begin{equation*} 
#   g'(\gamma) = \frac{ L_f^2}{2} - \frac{e_0^2}{2\gamma^2 T},
#   \quad
#   g''(\gamma) =  \frac{e_0^2}{\gamma^3 T} \geq 0
#   \end{equation*}
# minimal für
#   \begin{equation*} 
#   \gamma_{\min} = \frac{e_0}{L_f\sqrt{T}}
#   \end{equation*}
# mit
#   \begin{equation*} 
#   g_{\min} = g(\gamma_{\min}) = \frac{ L_f e_0}{\sqrt{T}}.
#   \end{equation*}

# Damit erhalten wir das folgende Ergebnis.
# 
# 
# **Satz:** $f:\mathbb{R}^d\to \mathbb{R}$, konvex, $C^1$, L-stetig mit Konstante $L_f$
# und es existiere $x_\ast = \mathrm{argmin}_{x\in\mathbb{R}^d}f(x)$.
# 
# Mit $\gamma = \frac{c}{T^\omega}$, $\omega\in(0,1)$, gilt
# \begin{align*} 
# \min_{t=0,\ldots,T-1}(f_t - f_\ast)
# &\leq \frac{1}{T} \sum_{t=0}^{T-1} (f_t - f_\ast)\\
# &= \mathcal{O}\Big(\big(\frac{1}{T}\big)^{\min(\omega,1-\omega)}\Big)
# \quad 
# \text{für}
# \quad
# T\to\infty.
# \end{align*}
# 
# Die optimale Ordnung ist $\frac{1}{2}$ bei $\omega=\frac{1}{2}$.
# 
# Mit $e_0 = \|x_0 - x_\ast\|_2$, $\gamma = \frac{e_0}{L_f\sqrt{T}}$ gilt außerdem
# \begin{equation*} 
# \min_{t=0,\ldots,T-1}(f_t - f_\ast)
# \leq \frac{1}{T} \sum_{t=0}^{T-1} (f_t - f_\ast)
# \leq \frac{ L_f e_0}{\sqrt{T}}.
# \end{equation*}  

# **Bemerkung:**
# 
# - $\min_{t=0,\ldots,T-1}(f_t - f_\ast) \leq \varepsilon$ gilt damit sicher, falls
#     \begin{equation*} 
#     \frac{ L_f e_0}{\sqrt{T}} \leq \varepsilon
#     \end{equation*}
#   bzw.
#     \begin{equation*} 
#     T \geq \big(\frac{e_0}{L_f \varepsilon}\big)^2
#     \end{equation*}
# 
# - für $\min_{t=0,\ldots,T-1}(f_t - f_\ast) \leq \varepsilon$ benötigen 
#   wir damit höchstens $\mathcal{O}(\frac{1}{\varepsilon^2})$ Schritte
# 
# - in der Praxis gibt man $\varepsilon$ vor, bestimmt $T$ und das zugehörige (feste) 
#     \begin{equation*} 
#     \gamma = \frac{e_0}{L_f\sqrt{T}}
#     \end{equation*}  und führt dann (maximal) $T-1$ Schritte des Verfahrens durch
# 
# - für $\varepsilon\to 0$ gilt $T\to\infty$ und 
#     \begin{equation*} 
#     \gamma = \frac{e_0}{L_f\sqrt{T}} \to 0
#     \end{equation*}

# ## $L$-Glattheit

# Ist $f$ konvex und $C^1$, dann gilt
# \begin{equation*} 
# f(y) \geq f(x) + f'(x)(y-x),
# \end{equation*}
# d.h. der Graph von $f$ verläuft oberhalb seiner Tangenten.
# Zur Abschätzung nach oben führen wir den folgenden Begriff ein.
# 
# 
# **Definition:**
# $f:\mathbb{R}^d \supset \mathrm{dom}(f) \to \mathbb{R}$ (nicht notwendig konvex),
# $X\subset \mathrm{dom}(f)$. 
# $f$ heißt $L$-glatt auf $X$ falls ein $L>0$ existiert, mit
# \begin{equation*} 
# f(y) \leq f(x) + f'(x)(y-x) + \frac{1}{2}L \|y-x\|_2^2
# \quad \forall x,y\in X.
# \end{equation*}
# 
#   
# **Bemerkung:** Ist $f$ $L$-glatt, so verläuft der Graph von $f$ unterhalb der
# quadratischen Approximation 
# \begin{equation*} 
# q_{L,x}(y) = f(x) + f'(x)(y-x) + \frac{1}{2}L\|y-x\|_2^2.
# \end{equation*}

# In[1]:


import sympy as sy
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

fsize = 12

x = sy.symbols('x')
f  = sy.Lambda(x, x*x/2 + 1/2)
f1 = sy.Lambda(x, f(x).diff(x))

L  = 3/4
x0 = -1

ql = sy.Lambda(x, f(x0) + f1(x0)*(x-x0) + L*(x-x0)**2)

ql = sy.lambdify(x, ql(x))
f  = sy.lambdify(x, f(x))

x = np.linspace(-2,2)
xmin = x.min()
xmax = x.max()
plt.plot(x, f(x) , 'g', label = '$f$')
plt.plot(x, ql(x), 'r', label = '$q_{L,x}$')
plt.axis('off')
#
tic = 0.2
plt.plot([xmin, xmax], [0,0], 'k')
#
plt.text(x0, -tic, '$x$', ha = 'center', va = 'top', fontsize=fsize)
plt.plot([x0,x0], [-tic,f(x0)], 'g:')
#
plt.legend(loc='lower center', ncol=3, fontsize=fsize)
plt.ylim(xmin, f(xmin));


# $L$-Glattheit ist eng verknüpft mit der  Lipschitz-Stetigkeit
# des Gradienten $f'$.
# 
# 
# **Lemma:**
# $f:\mathbb{R}^d \supset \mathrm{dom}(f) \to \mathbb{R}$ 
# differenzierbar (nicht notwendig konvex).
# Ist $f'$  Lipschitz-stetig, d.h.
# \begin{equation*} 
# \|f'(y) - f'(x)\| \leq L \|y-x\| \quad \forall x,y,
# \end{equation*}
# dann gilt
# \begin{equation*} 
# \big(f'(y)-f'(x)\big)(y-x) \leq L \|y-x\|^2.
# \end{equation*}
#   
# **Beweis:**
# Da $f'(x), f'(y)$ lineare stetige Operatoren sind gilt
# \begin{align*} 
# \big(f'(y)-f'(x)\big)(y-x)
# &\leq \big|\big(f'(y)-f'(x)\big)(y-x) \big|\\
# &\leq \|f'(y)-f'(x)\| \ \|y-x\|\\
# &\leq L \|y-x\|^2
# \end{align*}
# 
# $\square$

# **Lemma:**
# $f:\mathbb{R}^d \supset \mathrm{dom}(f) \to \mathbb{R}$ differenzierbar (nicht notwendig konvex),
# $\mathrm{dom}(f)$ konvex.
# Dann ist
# \begin{equation*} 
# \big(f'(y)-f'(x)\big)(y-x) \leq L \|y-x\|^2
# \quad \forall x,y
# \end{equation*}
# äquivalent zu
# \begin{equation*} 
# f(y) \leq f(x) + f'(x)(y-x) + \frac{1}{2} L \|y-x\|^2
# \quad \forall x,y.
# \end{equation*}

# **Beweis:**
# 
# "$\Rightarrow$"
# 
# - mit
#     \begin{equation*} 
#     g(t) = f\big(x+t(y-x)\big)
#     \end{equation*}
#   folgt
#     \begin{equation*} 
#     g'(t) = f'\big(x+t(y-x)\big)(y-x)
#     \end{equation*}    
#   und für $t>0$
#     \begin{align*}
#     g'(t) - g'(0)
#     & = \big( f'\big(x+t(y-x)\big) - f'(x) \big) (y-x) \\
#     & = \frac{1}{t} \big( f'\big(x+t(y-x)\big) - f'(x) \big) t(y-x) \\
#     &\leq \frac{1}{t}  L \|t(y-x)\|^2 \\
#     & = tL \|y-x\|^2 
#     \end{align*}
# 
# - damit erhalten wir
#     \begin{align*}
#     f(y)
#     &= g(1) \\
#     &= g(0) + \int_0^1g'(\tau)\:d\tau\\
#     &\leq f(x)  + \int_0^1 g'(0) + \tau L \|y-x\|^2  \:d\tau\\
#     & = f(x) +  f'(x)(y-x) + \frac{1}{2} L \|y-x\|^2
#     \end{align*}
# 
# "$\Leftarrow$"
# 
# - nach Voraussetzung ist
#     \begin{equation*} 
#     f(y) \leq f(x) + f'(x)(y-x) + \frac{1}{2} L \|y-x\|^2
#     \end{equation*}
#   bzw.
#     \begin{equation*} 
#     f(x) \leq f(y) + f'(y)(x-y) + \frac{1}{2} L \|x-y\|^2
#     \end{equation*}
# 
# - Addition der beiden Ungleichungen liefert
#     \begin{equation*} 
#     f(y) + f(x) \leq f(x) + f(y) + \big(f'(x)-f'(y)\big) (y-x) +  L \|y-x\|^2,
#     \end{equation*}
#   also
#     \begin{equation*} 
#     \big(f'(y)-f'(x)\big)(y-x) \leq L \|y-x\|^2
#     \end{equation*}
# 
# $\square$

# **Bemerkung:**
# Ist $f'$ Lipschitz-stetig, dann ist $f$  L-glatt.
#   
# Ist $f$ konvex, $\mathrm{dom}(f)=\mathbb{R}^d$ und existiert 
# ein $x_\ast \in \mathrm{dom}(f)$ mit $f(x_\ast)=\inf_{x}f(x)$,
# dann gilt auch die Umkehrung.

# **Lemma:**
# $f:\mathbb{R}^d  \to \mathbb{R}$ differenzierbar (nicht notwendig konvex) und
# es existiere $x_\ast$ mit 
# $f(x_\ast)=\inf_{x\in\mathbb{R}^d}f(x)$.
# Ist $f$ L-glatt mit Konstante $L$, dann gilt
# \begin{equation*} 
# \frac{1}{2L}\|f'(x)\|_2^2 \leq f(x) - f(x_\ast) \leq \frac{L}{2}\|x - x_\ast\|_2^2
# \end{equation*}
# und $f'$ ist Lipschitz-stetig mit Konstante $L$.

# **Beweis:**
# 
# - Abschätzung nach oben:
#   
#   - da $x_\ast$ globaler Minimierer ist muss $f'(x_\ast)=0$ sein und somit
#       \begin{align*} 
#       f(x) 
#       &\leq f(x_\ast) + f'(x_\ast)(x-x_\ast) + \frac{1}{2} L \|x-x_\ast\|_2^2\\
#       &= f(x_\ast) + \frac{1}{2} L \|x-x_\ast\|_2^2
#       \end{align*}
#     
# - Abschätzung nach unten: 
#   
#   - wir benutzen
#       \begin{align*} 
#       f(x_\ast) &= \inf_y f(y) \leq \inf_y u(y),
#       \\
#       u(y) &= f(x) + f'(x)(y-x) + \frac{1}{2} L \|y-x\|_2^2
#       \end{align*}
#     und minimieren die quadratische Funktion $u$
#   
#   - für die Ableitungen erhalten wir
#       \begin{equation*} 
#       u'(y) = f'(x) + L(y-x),
#       \quad
#       u''(y) = L I
#       \end{equation*}
# 
#   - $u$ ist (strikt) konvex mit globalem Minimierer $y_\ast$ mit
#       \begin{equation*} 
#       0 = u'(y_\ast) \quad \Leftrightarrow \quad y_\ast - x = -\frac{1}{L}f'(x)
#       \end{equation*}
#     und Minimum
#       \begin{align*}
#       u_\ast = u(y_\ast) 
#       &=  f(x) + f'(x)(y_\ast-x) + \frac{1}{2} L \|y_\ast-x\|_2^2\\
#       &=  f(x) - \frac{1}{L}\|f'(x)\|_2^2 + \frac{1}{2} L \|\frac{1}{L}f'(x)\|_2^2\\
#       &=  f(x) - \frac{1}{2L} \|f'(x)\|_2^2
#       \end{align*}
#       
#   - somit folgt
#       \begin{equation*} 
#       f(x_\ast) \leq f(x) - \frac{1}{2L} \|f'(x)\|_2^2
#       \end{equation*}
# 
# - Lipschitz-Stetigkeit von $f'$:
#  
#   - wir betrachten die Funktion
#       \begin{equation*} 
#       g(y) = f(y) - f'(x)y
#       \end{equation*}
#   
#   - mit $f$ ist auch $g$ konvex und differenzierbar mit Ableitung
#       \begin{equation*} 
#       g'(y) = f'(y) - f'(x)
#       \end{equation*}
#   
#   - damit ist $g'(x)=0$, also ist $y_\ast = x$ globales Minimum von $g$
#     
#   - außerdem folgt aus der $L$-Glattheit von $f$ für beliebiges $z$
#       \begin{align*}
#       g(y) &+ g'(y)(z-y) + \frac{1}{2} L \|z-y\|_2^2 = \\
#       &= f(y) - f'(x)y + \big( f'(y) - f'(x) \big)(z-y) + \frac{1}{2} L \|z-y\|_2^2 \\
#       & = f(y) + f'(y)(z-y) + \frac{1}{2} L \|z-y\|_2^2 
#           - f'(x)y  - f'(x) (z-y) \\
#       & \geq f(z) - f'(x) z \\
#       & = g(z)
#       \end{align*}    
#       so dass auch $g$ $L$-glatt ist
#   
#   - somit können wir die Abschätzung nach unten aus dem vorherigen Teil auf $g$ anwenden
#       und erhalten wegen $y_\ast=x$
#       \begin{align*}
#       \frac{1}{2L} \|f'(y) - f'(x)\|_2^2
#       &=  \frac{1}{2L}\|g'(y)\|_2^2 \\
#       &\leq g(y) - g(y_\ast)\\
#       & =   g(y) - g(x)\\
#       &= f(y) - f'(x)y - \big(f(x) - f'(x)x\big)\\
#       &= f(y) - f(x) - f'(x)(y-x),
#       \end{align*}
#       also
#       \begin{equation*} 
#       f(y) - f(x) - f'(x)(y-x) \geq \frac{1}{2L} \|f'(y) - f'(x)\|_2^2
#       \end{equation*}
#       bzw. durch vertauschen von $x$ und $y$
#       \begin{equation*} 
#       f(x) - f(y) - f'(y)(x-y) \geq \frac{1}{2L} \|f'(y) - f'(x)\|_2^2
#       \end{equation*}
# 
#   - durch Addition der beiden Ungleichungen erhalten wir
#       \begin{equation*} 
#       (f'(y) - f'(x))(y-x) \geq \frac{1}{L} \|f'(y) - f'(x)\|_2^2
#       \end{equation*}
#     und mit Cauchy-Schwartz
#       \begin{align*} 
#       \|f'(y) - f'(x)\|_2^2 
#       &\leq L \big(f'(y) - f'(x)\big)(x-y)\\
#       &\leq L \|f'(y) - f'(x)\|_2 \|x-y\|_2,
#       \end{align*}
#     so dass $f'$ Lipschitz-stetig mit Konstante $L$ ist
#   
#   $\square$

# **Bemerkung:** 
# Sei $f:\mathbb{R}^d\to \mathbb{R}$ konvex, $C^1$ und es existiere $x_\ast \in \mathrm{dom}(f)$ mit 
# $f(x_\ast)=\inf_{x}f(x)$. 
# Dann ist äquivalent:
# 
# - $f$ ist $L$-glatt mit Parameter $L$
#   
# - $\|f'(y)-f'(x)\|_2 \leq L \|y-x\|_2 \quad \forall x,y\in \mathbb{R}^d$
# 
# 
# $L$-Glattheit ist also unter diesen Voraussetzungen äquivalent dazu, 
# dass $f'$ Lipschitz-stetig mit Konstante $L$ ist.

# Folgende Operation erhalten die $L$-Glattheit:
# 
# - für  $i=1,\ldots,m$ seien $f_i:\mathbb{R}^d \supset \mathrm{dom}(f_i) \to \mathbb{R}$, $L$-glatt mit Parameter $L_i$ und $\lambda_i \geq 0$. Dann ist
#     \begin{equation*} 
#     f = \sum_{i=1}^m \lambda_i f_i
#     \end{equation*}
#   $L$-glatt mit Konstante
#     \begin{equation*} 
#     L = \sum_{i=1}^m \lambda_i L_i
#     \end{equation*}
#   über 
#     \begin{equation*} 
#     \mathrm{dom}(f) = \bigcap_{i=1}^m \mathrm{dom}(f_i)
#     \end{equation*}
#     
# - ist $f:\mathbb{R}^d \supset \mathrm{dom}(f) \to \mathbb{R}$ $L$-glatt mit Konstante $L$, $g:\mathbb{R}^m\to \mathbb{R}^d$ affin linear, d.h.
#   \begin{equation*} 
#   g(z) = Az + b,
#   \end{equation*}
#   dann ist $f\circ g$ auch $L$-glatt mit  
#   \begin{equation*} 
#   \tilde{L} = L \, \|A\|_2^2,
#   \quad
#   \mathrm{dom}(f\circ g)
#   =
#   \big\{z \ | z\in\mathbb{R}^m,\ g(z)\in \mathrm{dom}(f)\big\}
#   \end{equation*}

# Für $f$ konvex und $L$-glatt werden wir nun günstigere Konvergenzresultate für Gradient-Descent erhalten.
# Wir benutzen $L$-Glattheit mit $y=x_{t+1}$, $x=x_t$
#   \begin{equation*} 
#   f_{t+1} \leq f_{t}+f_{t}^{\prime}(x_{t+1}-x_{t})+\frac{L}{2}\|x_{t+1}- x_{t}\|_{2}^{2}.
#   \end{equation*}
# Mit $x_{t+1}-x_{t}=-\gamma  f_t$ folgt
#   \begin{equation*} 
#   f_{t+1} 
#   \leq f_{t}-\gamma\|f_{t}^{\prime}\|_{2}^{2}+\frac{L}{2} \gamma^{2}\|f_{t}^{\prime} \|_{2}^{2}
#   =f_{t}-\underbrace{\gamma\big(1-\frac{L}{2}\gamma\big)}_{=: \beta}\|f_{t}^{\prime}\|_2^{2}.
#   \end{equation*}
# Für $\gamma > 0$ ist $\beta > 0$ genau dann, wenn 
#   \begin{equation*} 
#   1-\frac{L}{2}\gamma > 0,
#   \end{equation*}
# also genau dann, wenn
#   \begin{equation*} 
#   0 < \gamma < \frac{2}{L}.
#   \end{equation*}

# Damit folgt
# 
# 
# **Descent-Lemma:** Ist $f:\mathbb{R}^d\to \mathbb{R}$ $L$-glatt, dann gilt 
#   \begin{equation*} 
#   f_{t+1} 
#   \leq f_{t}-\beta\|f_{t}^{\prime}\|_2^{2},
#   \quad
#   \beta = \gamma\big(1-\frac{\gamma L}{2}\big)
#   \end{equation*}
# und $\beta > 0$ falls $0 < \gamma < \frac{2}{L}$.
# 
# 
# **Bemerkung:** Für $f$ $L$-glatt und  $0 < \gamma < \frac{2}{L}$ fällt $f_t$
#   also *monoton*.

# Damit verschärfen wir jetzt unser Konvergenzresultat aus dem vorherigen Abschnitt.
# Nach den Vorüberlegungen gilt
#   \begin{equation*} 
#   \sum_{t=0}^{T-1} (f_t - f_\ast)
#   \leq  \frac{\gamma}{2} \sum_{t=0}^{T-1} \|f_{t}^{\prime}\|_{2}^{2}
#   + \frac{1}{2 \gamma} \|x_{0}-x_{*}\|_{2}^{2}
#   - \frac{1}{2 \gamma} \|x_{T}-x_{*}\|_{2}^{2}.
#   \end{equation*}
#   
# Aus dem Descent-Lemma folgt
#   \begin{equation*} 
#   \|f_{t}^{\prime}\|_2^{2} \leq \frac{1}{\beta}(f_{t}-f_{t+1})
#   \end{equation*}
# und somit
#   \begin{align*}
#   \sum_{t=0}^{T-1} (f_t - f_\ast)
#   &\leq  \frac{\gamma}{2\beta} \sum_{t=0}^{T-1}(f_{t}-f_{t+1})
#   + \frac{1}{2 \gamma} \big(\|x_{0}-x_{*}\|_{2}^{2} - \|x_{T}-x_{*}\|_{2}^{2}\big)
#   \\
#   &=  \frac{\gamma}{2\beta} (f_{0}-f_{T}) 
#   + \frac{1}{2 \gamma} \big(\|x_{0}-x_{*}\|_{2}^{2} - \|x_{T}-x_{*}\|_{2}^{2}\big).
#   \end{align*}
# Für $\gamma < \frac{2}{L}$ ist $f_{t+1}\leq f_t$, so dass
#   \begin{equation*} 
#   f_{T-1} \leq \frac{1}{T}\sum_{t=0}^{T-1}f_t 
#   \end{equation*}
# und damit
#   \begin{align*} 
#   f_{T-1}-f_\ast 
#   &\leq \frac{1}{T}\sum_{t=0}^{T-1}(f_t -f_\ast)\\
#   &\leq \frac{1}{T}
#   \Big(
#   \frac{\gamma}{2\beta} (f_{0}-f_{T}) 
#   + \frac{1}{2 \gamma} \big(\|x_{0}-x_{*}\|_{2}^{2} - \|x_{T}-x_{*}\|_{2}^{2}\big) 
#   \Big).
#   \end{align*}
# Mit $\|x_{T}-x_{*}\|_{2}\geq 0$ erhalten wir schließlich
#   \begin{equation*} 
#   f_{T-1}-f_\ast 
#     \leq \frac{1}{T}
#   \Big(
#   \frac{\gamma}{2\beta} (f_{0}-f_{\ast}) 
#   + \frac{1}{2 \gamma} \|x_{0}-x_{*}\|_{2}^{2} 
#   \Big)
#   \end{equation*}

# **Satz:** $f:\mathbb{R}^d\to \mathbb{R}$, konvex, $L$-glatt mit Konstante $L$
# und es existiere $x_\ast = \mathrm{argmin}_{x\in\mathbb{R}^d}f(x)$.
# Für  $0 < \gamma < \frac{2}{L}$ ist
# \begin{align*} 
# f_{T}-f_\ast 
# &\leq \frac{1}{T+1}
# \Big(
# \frac{\gamma}{2\beta} (f_{0}-f_{\ast}) 
# + \frac{1}{2 \gamma} \|x_{0}-x_{*}\|_{2}^{2} 
# \Big)
# \\
# &= \mathcal{O}\big( \frac{1}{T} \big)
# \quad \text{für}\quad  T\to\infty.
# \end{align*}

# **Bemerkung:**
# 
# - $f_{T} - f_\ast \leq \varepsilon$ gilt damit sicher, falls
#     \begin{equation*} 
#     \frac{1}{T+1}
#     \Big(
#     \frac{\gamma}{2\beta} (f_{0}-f_{\ast}) 
#     + \frac{1}{2 \gamma} \|x_{0}-x_{*}\|_{2}^{2} 
#     \Big) \leq \varepsilon
#     \end{equation*}
#   also
#     \begin{equation*} 
#     T \geq \frac{1}{\varepsilon}
#     \Big(
#     \frac{\gamma}{2\beta} (f_{0}-f_{\ast}) 
#     + \frac{1}{2 \gamma} \|x_{0}-x_{*}\|_{2}^{2} 
#     \Big) - 1
#     = \mathcal{O}\big(\frac{1}{\varepsilon}\big)
#     \end{equation*}
#   
# - da $\gamma < \frac{2}{L}$ sein muss kann wegen des Terms
#     \begin{equation*} 
#     \frac{1}{2 \gamma} \|x_{0}-x_{*}\|_{2}^{2}
#     \end{equation*}
#   die Asymptotik der oberen Schranke nicht mehr durch
#   eine $T$-abhängige Wahl von $\gamma$ verbessert werden

# ## $\mu$-Konvexität

# Bis jetzt haben wir nur Abschätzungen für $f_t - f_\ast$ bewiesen.
# Nun werden wir $\|x_t - x_\ast\|_2$ betrachten.
# Dazu benötigen wir nochmals stärkere Voraussetzungen, nämlich $\mu$-Konvexität.
# Damit werden wir zusätzlich auch eine besseres asymptotisches verhalten
# nachweisen können.

# $f$ ist $\mu$-konvex falls $f(x)-\frac{\mu}{2}\|x\|_2^2$ konvex ist.
# Ist $f$ zusätzlich differenzierbar, so erhalten wir das folgende Ergebnis
# 
# 
# **Lemma:** Ist $f$ $\mu$-konvex und differenzierbar, dann gilt
#   \begin{equation*} 
#   f(y) \geq f(x) + f'(x)(y-x) + \frac{\mu}{2}\|y-x\|_2^2
#   \quad
#   \forall x,y
#   \end{equation*}
# und
#   \begin{equation*} 
#   \big(f'(y) - f'(x)\big)(y-x) \geq \mu \|y-x\|_2^2
#   \quad
#   \forall x,y.
#   \end{equation*}

# **Beweis:**
# 
# - $f$ ist $\mu$-konvex falls $g(x) = f(x)-\frac{\mu}{2}\|x\|_2^2$ konvex ist
#   
# - mit $f$ ist auch $g$ differenzierbar mit
#     \begin{equation*} 
#     g'(x) = f'(x) - \mu x
#     \end{equation*}
#   
# - wegen
#     \begin{equation*} 
#     g(y) \geq g(x)+g'(x)(y-x)
#     \end{equation*}
#   ist
#     \begin{equation*} 
#     f(y)-\frac{\mu}{2}\|y\|_{2}^{2} 
#     \geq 
#     f(x)-\frac{\mu}{2}\|x\|_2^{2} + \big(f'(x) - \mu x\big)^T(y-x)
#     \end{equation*}
#   bzw.
#     \begin{equation*} 
#     f(y) 
#     \geq f(x)+f'(x)(y-x)
#     +\frac{\mu}{2} \big(\underbrace{y^{T} y-x^{T} x-2 x^{T}(y-x)}_{h}\big)
#     \end{equation*}
#   mit
#     \begin{align*} 
#     h 
#     &= y^{T} y - x^{T} x - 2 x^{T}y + 2 x^{T}x
#     \\
#     &= y^{T} y - 2 x^{T}y + x^{T} x
#     \\
#     &= \|y-x\|_2^2,
#     \end{align*}
#   also
#     \begin{equation*} 
#     f(y) \geq f(x) + f'(x)(y-x) + \frac{\mu}{2}\|y-x\|_2^2
#     \end{equation*}
#     
#   da $g$ konvex und differenzierbar ist, ist $g'$ monoton,
#   also
#     \begin{align*}
#     0 
#     & \leq \big(g'(y) - g'(x)\big)(y-x) \\
#     &=  \big(f'(y) - \mu y - f'(x) + \mu x\big)(y-x)\\
#     &= \big(f'(y) - f'(x)\big)(y-x) - \mu \|y-x\|_2^2
#     \end{align*}
#   und somit
#     \begin{equation*} 
#     \big(f'(y) - f'(x)\big)(y-x) \geq \mu \|y-x\|_2^2
#     \quad
#     \forall x,y
#     \end{equation*}
#     
# $\square$

# **Bemerkung:**
# 
# - ist $f$ $\mu$-konvex und $L$-glatt (und damit differenzierbar), so gilt
#     \begin{align*}
#     f(y) &\leq f(x) + f'(x)(y-x) + \frac{L}{2}\|y-x\|_2^2 =: q_{L,x}(y) \\
#     f(y) &\geq f(x) + f'(x)(y-x) + \frac{\mu}{2}\|y-x\|_2^2 =: q_{\mu,x}(y)
#     \end{align*}
#   
# - $f$ kann also zwischen den beiden quadratischen Funktionen $q_{\mu,x}$, $q_{L,x}$
#   "eingesperrt" werden
#   
# - $q_{\mu,x}$, $q_{L,x}$ berühren $f$ im Punkt $x$
# 
# - es muss immer $\mu \leq L$ gelten
#   
# - ist $\mu>0$ so ist $f$ strikt konvex und $x_\ast$ ist damit eindeutig
#   
# - ist $\mu = L$ dann ist $f = q_{\mu,x} = q_{L,x}$, d.h. $f$ ist ein quadratisches
#   Polynom

# In[2]:


import sympy as sy
import numpy as np
import matplotlib.pyplot as plt

fsize = 12

x = sy.symbols('x')
f  = sy.Lambda(x, x*x/2 + 1/2)
f1 = sy.Lambda(x, f(x).diff(x))

m  = 1/3
L  = 3/4
x0 = -1

qm = sy.Lambda(x, f(x0) + f1(x0)*(x-x0) + m*(x-x0)**2)
ql = sy.Lambda(x, f(x0) + f1(x0)*(x-x0) + L*(x-x0)**2)


x1 = sy.solve(ql(x).diff(x))[0]
x2 = sy.solve(f1(x))[0]
x3 = sy.solve(qm(x).diff(x))[0]              

qm = sy.lambdify(x, qm(x))
ql = sy.lambdify(x, ql(x))
f  = sy.lambdify(x, f(x))

x = np.linspace(-2,3)
xmin = x.min()
xmax = x.max()
plt.plot(x, f(x) , 'g', label = '$f$')
plt.plot(x, qm(x), 'b', label = '$q_{\mu,x}$')
plt.plot(x, ql(x), 'r', label = '$q_{L,x}$')
plt.axis('off')
#
tic = 0.2
plt.plot([xmin, xmax], [0,0], 'k')
#
plt.text(x2, -tic, '$x_{*}$', ha = 'center', va = 'top', fontsize=fsize)
plt.plot([x2,x2], [-tic,f(x2)], 'g:')
#
plt.text(x0, -tic, '$x$', ha = 'center', va = 'top', fontsize=fsize)
plt.plot([x0,x0], [-tic,f(x0)], 'g:')
plt.plot([xmin,x0], [f(x0),f(x0)], 'g:')
#
plt.plot([xmin,x1], [ql(x1),ql(x1)], 'r:')
#
plt.text(xmin-tic, (f(x0)+f(x1)+tic)/2, 'guaranteed progress', color='r', ha = 'center', va = 'center', fontsize=fsize)
#
#
plt.plot([x0,xmax], [f(x0),f(x0)], 'g:')
plt.plot([x3,xmax], [qm(x3),qm(x3)], 'b:')
plt.text(xmax, (f(x0)+qm(x3))/2, 'maximal\n suboptimality', color='b', ha = 'center', va = 'center', fontsize=fsize)
#
plt.legend(loc='lower center', ncol=3, fontsize=fsize)
plt.ylim(xmin, f(xmin));


# **Lemma:**
# $f:\mathbb{R}^d  \to \mathbb{R}$ differenzierbar und
# es existiere $x_\ast$ mit 
# $f(x_\ast)=\inf_{x\in\mathbb{R}^d}f(x)$.
# Ist $f$ $\mu$-konvex mit Konstante $\mu$, dann gilt
# \begin{equation*} 
# \frac{\mu}{2}\|y-x\|_2^2\leq f(x) - f(x_\ast) \leq \frac{1}{2\mu} \|f'(x)\|_2^2. 
# \end{equation*}

# **Beweis:**
# 
# - $f$ ist $\mu$-konvex, d.h.
#     \begin{equation*} 
#     f(y) \geq f(x) + f'(x)(y-x) + \frac{\mu}{2}\|y-x\|_2^2  \quad \forall x,y
#     \end{equation*}
# 
# - Abschätzung nach unten:
#   
#   - da $x_\ast$ globaler Minimierer ist muss $f'(x_\ast)=0$ sein und 
#     aus der $\mu$-Konvexität folgt
#       \begin{align*} 
#       f(x) 
#       &\geq f(x_\ast) + f'(x_\ast)(x-x_\ast) + \frac{\mu}{2}  \|x-x_\ast\|_2^2\\
#       &   = f(x_\ast) + \frac{\mu}{2} \|x-x_\ast\|_2^2
#       \end{align*}
#     
# - Abschätzung nach oben: 
#   
#   - wir benutzen
#       \begin{align*} 
#       f(x_\ast) &= \inf_y f(y) \geq \inf_y u(y),
#       \\
#       u(y) &= f(x) + f'(x)(y-x) + \frac{\mu}{2}  \|y-x\|_2^2
#       \end{align*}
#     und minimieren die quadratische Funktion $u$
#   
#   - für die Ableitungen erhalten wir
#       \begin{equation*} 
#       u'(y) = f'(x) + \mu (y-x),
#       \quad
#       u''(y) = \mu I
#       \end{equation*}
# 
#   - für $\mu>0$ ist $u$ strikt konvex mit globalem Minimierer $y_\ast$ mit
#       \begin{equation*} 
#       0 = u'(y_\ast) \quad \Leftrightarrow \quad y_\ast - x = -\frac{1}{\mu}f'(x)
#       \end{equation*}
#     und Minimum
#       \begin{align*}
#       u_\ast = u(y_\ast) 
#       &=  f(x) + f'(x)(y_\ast-x) + \frac{1}{2} \mu \|y_\ast-x\|_2^2\\
#       &=  f(x) - \frac{1}{\mu} \|f'(x)\|_2^2 + \frac{1}{2} \mu \|\frac{1}{\mu}f'(x)\|_2^2\\
#       &=  f(x) - \frac{1}{2\mu} \|f'(x)\|_2^2
#       \end{align*}
#       
#   - somit folgt
#       \begin{equation*} 
#       f(x_\ast) \geq  f(x) - \frac{1}{2\mu} \|f'(x)\|_2^2.
#       \end{equation*}
#     bzw.
#       \begin{equation*} 
#       f(x) - f(x_\ast) \leq \frac{1}{2\mu} \|f'(x)\|_2^2 
#       \end{equation*}
#       
# $\square$

# Aus $x_{t+1} = x_t - \gamma f'_t$ hatten wir in den Vorüberlegungen
#   \begin{equation*} 
#   \|x_{t+1}-x_{*}\|_{2}^{2} 
#   =\|x_{t}-x_{*}\|_{2}^{2}+\gamma^{2}\|f_{t}^{\prime}\|_{2}^{2}-2
#   \gamma f_{t}^{\prime}(x_{t}-x_{*})
#   \end{equation*}
# erhalten.
# $\mu$-Konvexität liefert mit $y=x_\ast$, $x=x_t$
#   \begin{equation*} 
#   f_\ast \geq f_t + f'_t(x_\ast-x_t) + \frac{\mu}{2} \|x_\ast-x_t\|_2^2
#   \end{equation*}
# bzw.
#   \begin{equation*} 
#   -f'_t(x_t - x_\ast)  \leq  f_\ast - f_t - \frac{\mu}{2} \|x_t - x_\ast\|_2^2.
#   \end{equation*}
# Eingesetzt erhalten wir
#   \begin{align*}
#   \|x_{t+1}-x_{*}\|_{2}^{2} 
#   & =\|x_{t}-x_{*}\|_{2}^{2}+\gamma^{2}\|f_{t}^{\prime}\|_{2}^{2}
#      + 2 \gamma \big( f_\ast - f_t - \frac{\mu}{2} \|x_t - x_\ast\|_2^2 \big)\\
#   & \leq
#   (1-\gamma\mu) \|x_{t}-x_{*}\|_{2}^{2}+\gamma^{2}\|f_{t}^{\prime}\|_{2}^{2}
#    + 2 \gamma (f_\ast - f_t).
#   \end{align*}

# Das Descent-Lemma aus dem vorherigen Kapitel liefert
#   \begin{equation*} 
#   f_\ast - f_t 
#   \leq f_{t+1} - f_{t}
#   \leq -\beta\|f_{t}^{\prime}\|_2^{2},
#   \quad
#   \beta = \gamma\big(1-\frac{\gamma L}{2}\big)
#   \end{equation*}
# mit $\beta > 0$ für $0 < \gamma < \frac{2}{L}$, so dass
#   \begin{equation*} 
#   \|x_{t+1}-x_{\ast}\|_{2}^{2}
#   \leq 
#   (1-\gamma\mu) \ \|x_{t}-x_{*}\|_{2}^{2}
#   +(\gamma^{2} -  2 \gamma \beta)  \ \|f_{t}^{\prime}\|_{2}^{2}
#   \end{equation*}
# gilt.
#   
# Für den Vorfaktor des letzten Terms gilt wegen $\gamma>0$, $\beta = \gamma\big(1-\frac{\gamma L}{2}\big)$,
#   \begin{align*}
#   \gamma (\gamma - 2  \beta) \leq 0
#   & \quad\Leftrightarrow \quad \gamma \leq 2  \beta = \gamma(2 - \gamma L) \\
#   & \quad\Leftrightarrow \quad 1 \leq 2 - \gamma L \\
#   & \quad\Leftrightarrow \quad \gamma \leq \frac{1}{L} .
#   \end{align*}
# Somit folgt für $0<\gamma \leq \frac{1}{L}$
#   \begin{equation*} 
#   \|x_{t+1}-x_{*}\|_{2}^2 \leq \rho \|x_{t}-x_{*}\|_{2}^2,
#   \quad
#   \rho = 1-\gamma\mu 
#   \end{equation*}
# bzw.
#   \begin{equation*} 
#   \|x_{T}-x_{*}\|_{2}^2 \leq \rho^T \|x_{0}-x_{*}\|_{2}^2.
#   \end{equation*}
# Wegen $0<\mu\leq L$ und $0<\gamma\leq \frac{1}{L}$ ist 
#   \begin{equation*} 
#   0<\gamma\mu\leq \gamma L \leq 1
#   \end{equation*} 
# und somit
#   \begin{equation*} 
#   0 \leq \rho < 1,
#   \end{equation*}
# also $|\rho|< 1$ und wir erhalten Konvergenz für $x_T$.
# 
# Für $f_T$ ergibt sich direkt aus der $L$-Glattheit
#   \begin{equation*} 
#   f_T \leq f_\ast + f'_\ast(x_T - x_\ast) + \frac{L}{2} \|x_T - x_\ast\|_2^2,
#   \end{equation*}
# und wegen $f'_\ast=0$
#   \begin{equation*} 
#   f_T - f_\ast  
#   \leq \frac{L}{2} \|x_T - x_\ast\|_2^2
#   \leq \frac{L}{2} \rho^T \|x_{0}-x_{*}\|_{2}^2.
#   \end{equation*}  

# Insgesamt haben wir damit das folgende Ergebnis bewiesen.
# 
# 
# **Satz:** $f:\mathbb{R}^d\to \mathbb{R}$, $\mu$-konvex mit $\mu>0$, $L$-glatt mit Konstante $L$
# und es existiere $x_\ast = \mathrm{argmin}_{x\in\mathbb{R}^d}f(x)$.
#   
# Für  $0 < \gamma \leq \frac{1}{L}$ folgt
# \begin{equation*} 
# \|x_{t+1}-x_{*}\|_{2}^2 \leq \rho \|x_{t}-x_{*}\|_{2}^2
# \end{equation*}
# und
# \begin{equation*} 
# f_T - f_\ast \leq \frac{L}{2} \rho^T \|x_{0}-x_{*}\|_{2}^2
# \end{equation*}
# mit
# \begin{equation*} 
# \rho = 1-\gamma\mu \in [0,1).
# \end{equation*}

# **Bemerkung:**
# 
# - $f_{T} - f_\ast \leq \frac{L}{2} \rho^T \|x_{0}-x_{*}\|_{2}^2 \leq\varepsilon$ 
#   gilt sicher, falls
#     \begin{equation*} 
#     \rho^T \leq \frac{2\varepsilon}{L \|x_{0}-x_{*}\|_{2}^2}
#     \end{equation*}
# 
# - für $\rho = 0$ gilt das für alle $\varepsilon\ge 0$
#   
# - für $0 < \rho < 1$  folgt
#     \begin{equation*} 
#     T\log(\rho) \leq \log\big(\frac{2\varepsilon}{L \|x_{0}-x_{*}\|_{2}^2}\big), 
#     \quad \log(\rho)<0
#     \end{equation*}
#   also
#     \begin{align*}
#     T 
#     &\geq 
#     \frac{1}{ \log(\rho) } \log\big(\frac{2\varepsilon}{L \|x_{0}-x_{*}\|_{2}^2}\big)\\
#     &=
#     \frac{1}{|\log(\rho)|} \log\big(\frac{L \|x_{0}-x_{*}\|_{2}^2}{2\varepsilon}\big)\\
#     &=
#     \frac{1}{|\log(\rho)|}
#     \Big(
#     \log\big(\frac{L}{2} \|x_{0}-x_{*}\|_{2}^2\big)
#     +
#     \log\big(\frac{1}{\varepsilon}\big)
#     \Big)
#     \end{align*}
#   und somit
#     \begin{equation*} 
#     T = \mathcal{O}\Big( \log\big(\frac{1}{\varepsilon}\big) \Big)
#     \end{equation*}

# ## Zusammenfassung

# Für Gradient-Descent bei *nicht restringierten Optimierungsproblemen* haben wir folgendes 
#   Konvergenzverhalten nachgewiesen:
# 
# - $f$ konvex und Lipschitz-stetig, $\gamma = \frac{c}{\sqrt{T}}$, $c>0$:
#     \begin{equation*} 
#     \min_{t=0,\ldots,T-1}(f_t - f_\ast) \leq \varepsilon
#     \quad \Rightarrow\quad
#     T = \mathcal{O}\big(\frac{1}{\varepsilon^2}\big)
#     \end{equation*}
#     
# - $f$ konvex und $L$-glatt, $0 < \gamma < \frac{2}{L}$:
#     \begin{equation*} 
#     f_{T}-f_\ast \leq \varepsilon
#     \quad \Rightarrow\quad
#     T =\mathcal{O}\big( \frac{1}{\varepsilon} \big)
#     \end{equation*}  
# 
# - $f$ $\mu$-konvex mit $\mu>0$ und $L$-glatt, $0 < \gamma \leq \frac{1}{L}$:
#     \begin{equation*} 
#     f_{T}-f_\ast \leq \varepsilon
#     \quad \Rightarrow\quad
#     T = \mathcal{O}\Big( \log\big(\frac{1}{\varepsilon}\big) \Big)
#     \end{equation*}
#     
# Ist $f(x)$ eine $L$-glatte Funktion, dann ist für alle $\mu > 0$
#   \begin{equation*} 
#   f_R(x) = f(x) + \mu \|x\|_2^2
#   \end{equation*}
# auch $\mu$-konvex, d.h. Tikhonov(Ridge)-Regularisierung kann die
# Konvergenz von Gradient-Descent beschleunigen.
