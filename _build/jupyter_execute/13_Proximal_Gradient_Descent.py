#!/usr/bin/env python
# coding: utf-8

# # Proximal Gradient Descent

# ## Überblick

# Bei Subgradient-Descent haben wir gesehen, dass fehlende Differenzierbarkeit von $f$ kein Problem ist, solange Subgradienten existieren.
# Allerdings reduziert sich bei fehlender Glattheit die Konvergenzgeschwindigkeit.
# 
# Hat $f$ zusätzlich eine spezielle Struktur, so kann man Gradient-Descent so modifizieren, dass man Konvergenzraten wie im $C^1$-Fall erhält.
# 
# Wir untersuchen hier nicht restringierte Probleme, restringierte lassen sich bei
#   Proximal-Gradient-Descent
#   durch geeignete Wahl der Zielfunktion auf restringierte zurück führen.

# ## Grundlagen

# Für $\gamma > 0$ und festes $z$ ist die Funktion
#   \begin{equation*} 
#   q(y) = \frac{1}{2\gamma} \|y - z\|_2^2
#   \end{equation*}
# strikt konvex mit globalem Minimum $y_\ast=z$.

# Für $f$ differenzierbar kann man deshalb einen Gradient-Descent-Schritt
#   \begin{equation*} 
#   x_{t+1} = x_t - \gamma f'_t
#   \end{equation*}
# auch schreiben kann als
#   \begin{equation*} 
#   x_{t+1} 
#   = 
#   \underset{y}{\mathrm{argmin}} 
#   \Big(
#   \frac{1}{2\gamma}
#   \big\|
#   y - (x_t - \gamma f'_t)
#   \big\|_2^2
#   \Big).
#   \end{equation*}

# Wir nehmen nun an, dass $f=g+h$ ist mit
# 
# - $g$ differenzierbar, $L$-glatt und $\mu$-konvex
#   
# - $h$ "nur" konvex
#   
# 
# **Beispiel:** Bei Lasso ist
#   \begin{equation*} 
#   f(w) = \underbrace{\frac{1}{2m}\|Xw-y\|_2^2}_{g(w)}
#          + \underbrace{\alpha\|w\|_1}_{h(w)}.
#   \end{equation*}

# Wir benutzen jetzt die Darstellung von Gradient-Descent von oben,
#   behandeln $g$ "wie üblich" und hängen $h$ "unbehandelt" an
#   \begin{align*}
#   x_{t+1} 
#   &= 
#   \underset{y}{\mathrm{argmin}} 
#   \Big(
#   \frac{1}{2\gamma}
#   \big\|
#   y - (x_t - \gamma g'_t)
#   \big\|_2^2
#   + h(y)
#   \Big)\\
#   &= 
#   \underset{y}{\mathrm{argmin}} 
#   \Big(
#   \frac{1}{2}
#   \big\|
#   y - (x_t - \gamma g'_t)
#   \big\|_2^2
#   + \gamma h(y)
#   \Big).
#   \end{align*}
#   
# Das führt uns zu folgender Notation.
# 
# 
# **Definition:**
# 
# - für $h$ konvex, $\gamma > 0$ ist 
#     der *Proximal-Operator* definiert als
#     \begin{equation*} 
#     \mathrm{prox}(x) = 
#       \underset{y}{\mathrm{argmin}} 
#       \big(
#       \frac{1}{2} \|y - x\|_2^2 + \gamma h(y)
#       \big)
#     \end{equation*}
#     
# - für $f=g+h$, $g$ differenzierbar, $h$ konvex, ist *Proximal-Gradient-Descent* gegeben durch
#   \begin{equation*} 
#   x_{t+1} = \mathrm{prox}(y_{t+1}),
#   \quad
#   y_{t+1} = x_t - \gamma g'_t
#   \end{equation*}

# **Bemerkung:**
# 
# - man kann zeigen, dass für
#     $h$ konvex, halbstetig von unten, "proper" ($h(x)>-\infty$ $\forall x$,
#     $h$ nicht identisch $+\infty$)
#     der Proximal-Operator  wohldefiniert ist
#     (Eindeutigkeit ist klar, da
#     $\frac{1}{2} \|y - x\|_2^2 + \gamma h(y)$ strikt konvex ist, die sonstigen
#     Voraussetzungen an $h$ braucht man zum Nachweis der Existenz)
#   
# - Proximal-Gradient-Descent ähnelt Projected-Gradient-Descent, wobei 
#     $\Pi$ durch $\mathrm{prox}$ ersetzt wird 
#     (Übung: Projected-Gradient-Descent ist Spezialfall von 
#     Proximal-Gradient-Descent)

# ## Konvergenz

# Die Konvergenzbeweise verlaufen ähnlich wie bei Projected-Gradient-Descent. Dazu benötigen wir Informationen über die Abbildungseigenschaften von $\mathrm{prox}(\cdot)$. 
#   
# 
# **Lemma:**
# 
#   - ist $x_\ast = \underset{x}{\mathrm{argmin}}  f(x)$, dann gilt
#     $x_\ast = \mathrm{prox}(x_\ast - \gamma g'_\ast)$ 
#     (Übung)
#     
#   - $\|\mathrm{prox}(y) - \mathrm{prox}(x)\|_2 \leq \|y-x\|_2 \quad\forall x,y$

# Es sei nun $g$ differenzierbar, $L$-glatt und $\mu$-konvex.
# Wegen der $L$-Glattheit ist $g'$ Lipschitz-stetig, d.h.
#   \begin{equation*} 
#   \|g'(y) - g'(x)\|_2 \leq L \|y -x\|_2 .
#   \end{equation*}
# 
# $\mu$-Konvexität bedeutet
#   \begin{equation*} 
#   g(y) \geq g(x) + g'(x)(y-x) + \frac{\mu}{2}\|y -x\|_2^2    
#   \quad \forall x,y,
#   \end{equation*}
# also auch
#   \begin{equation*} 
#   g(x) \geq g(y) + g'(y)(x-y) + \frac{\mu}{2}\|x -y\|_2^2   
#   \quad \forall x,y.
#   \end{equation*}  
# Addition der beiden Ungleichungen liefert
#   \begin{equation*} 
#   g(y) + g(x) \geq g(x) + g(y) + (g'(x)-g'(y))(y-x) + \mu \|y -x\|_2^2 ,
#   \end{equation*}
# so dass 
#   \begin{equation*} 
#   \big(g'(y)-g'(x)\big)(y-x) 
#   \geq   \mu \|y -x\|_2^2  .
#   \end{equation*}
#   
# Mit Cauchy-Schwartz folgt
#   \begin{equation*} 
#   \mu \|y -x\|_2^2  \leq \big(g'(y)-g'(x)\big)(y-x) \leq \|g'(y) - g'(x)\|_2\|y -x\|_2
#   \end{equation*}
#   und schließlich
#   \begin{equation*} 
#   \|g'(y) - g'(x)\|_2 \geq \mu \|y -x\|_2 .  
#   \end{equation*}

# Für $\beta\in\mathbb{R}$ beliebig erhalten wir
#   \begin{equation*} 
#   \beta\|g'(y) - g'(x)\|_2^2 \leq \beta L^2 \|y -x\|_2^2 \quad\text{für}\quad \beta\geq 0 
#   \end{equation*}
# bzw.
#   \begin{equation*} 
#   \beta\|g'(y) - g'(x)\|_2^2 \leq \beta \mu^2 \|y -x\|_2^2 \quad\text{für}\quad \beta < 0 
#   \end{equation*}  
# und somit
#   \begin{equation*} 
#   \beta\|g'(y) - g'(x)\|_2^2 
#   \leq \max(\beta L^2, \beta\mu^2) \|y -x\|_2^2 
#   \quad \forall \beta \in \mathbb{R}.
#   \end{equation*}

# Außerdem ist
#   \begin{align*}
#   \big(g'(y)-g'(x)\big)(y-x) 
#   &\geq   \mu \|y -x\|_2^2  \\
#   &= \frac{L}{L+\mu} \mu \|y -x\|_2^2 + \frac{\mu}{L+\mu} \mu \|y -x\|_2^2 \\
#   &\geq \frac{L}{L+\mu} \mu \|y -x\|_2^2 
#       + \frac{\mu^2}{L+\mu} \frac{1}{L^2}\|g'(y) - g'(x)\|_2^2 .
#   \end{align*}
# Wegen $0 < \mu \leq L$ folgt 
#   \begin{equation*} 
#   \big(g'(y)-g'(x)\big)(y-x) 
#   \geq
#   \frac{1}{L +\mu} \|g'(y) - g'(x)\|_2^2
#   +\frac{L\mu}{L+\mu} \|y -x\|_2^2. 
#   \end{equation*}
#   
# Damit erhalten wir das folgende Resultat.

# **Satz:** Ist $f=g+h$, $g$ differenzierbar, $L$-glatt und $\mu$-konvex,
#   $h$ konvex, $x_\ast = \mathrm{argmin}_x f(x)$ und
#   \begin{equation*}  
#   \ x_{t+1} = \mathrm{prox}(x_t - \gamma g'_t),
#   \end{equation*}
#   dann gilt
#   \begin{equation*} 
#   \|x_t - x_\ast\|_2 \leq Q(\gamma)^t \|x_0 - x_\ast\|_2
#   \end{equation*}
#   mit
#   \begin{equation*} 
#   Q(\gamma) = \max(|1-\gamma L|, |1- \gamma \mu|).
#   \end{equation*}

# **Beweis:**
# 
# - es ist
#    \begin{align*}
#    \|x_{t+1} - x_\ast\|_2^2
#    & = \|  \mathrm{prox}(x_t - \gamma g'_t) 
#          - \mathrm{prox}(x_\ast - \gamma g'_\ast)\|_2^2 \\
#    & \leq \|x_{t}-\gamma g_{t}'-(x_\ast-\gamma g'_\ast)\|_{2}^{2} \\
#    &   =  \|x_{t}-x_\ast-\gamma(g'_{t}-g'_\ast)\|_{2}^{2} \\
#    &   =  \|x_{t}-x_\ast\|_{2}^{2}
#           +\gamma^{2}\|g_{t}'-g'_\ast\|_{2}^{2} 
#           -2 \gamma(g'_{t}-g'_\ast) (x_{t}-x_\ast)\\
#    & \leq \|x_{t}-x_\ast\|_{2}^{2}
#           +\gamma^{2}\|g_{t}'-g'_\ast\|_{2}^{2} \\
#    & \quad-2 \gamma
#              \big(
#              \frac{1}{L+\mu}\|g_{t}'-g_\ast'\|_{2}^{2}
#              +\frac{L {\mu}}{L+\mu}\|x_{t}-x_{\ast}\|_{2}^{2}
#              \big) \\
#    & =  \underbrace{\Big(1-\frac{2 \gamma L \mu}{L+\mu}\Big)
#         }_{\alpha}
#         \|x_{t}-x_\ast\|_{2}^{2}
#        +\underbrace{
#           \gamma \Big(\gamma-\frac{2}{L+\mu}\Big)
#          }_{\beta}
#          \|g_{t}'-g'_\ast\|_{2}^{2}
#    \end{align*}
#   und somit
#    \begin{align*}
#    \|x_{t+1} - x_\ast\|_2^2 
#    & \leq \alpha \|x_t - x_\ast\|_2^2 
#         + \max\big(\beta L^2, \beta \mu^2\big) \|x_t - x_\ast\|_2^2 \\
#    &  =  \max\big(\alpha + \beta L^2, \alpha + \beta \mu^2\big) \|x_t - x_\ast\|_2^2 
#    \end{align*}
#    
# - mit 
#     \begin{align*}
#     \alpha+\beta L^{2} 
#     &=1-\frac{2 \gamma L{\mu}}{L+\mu}+\gamma\big(\gamma-\frac{2}{L+\mu}\big) L^{2} \\
#     &=1+\gamma^{2} L^{2}-\frac{1}{L+\mu}\big(2 \gamma L \mu+2 \gamma L^{2}\big) \\
#     &=1+\gamma^{2} L^{2}-\frac{2}{L+\mu} \gamma L(L+\mu) \\
#     &=1+\gamma^{2} L^{2}-2 \gamma L \\
#     &=(1-\gamma L)^{2}
#     \end{align*}
#   und
#     \begin{align*}
#     \alpha+\beta \mu^{2} 
#     &=1-\frac{2 \gamma L{\mu}}{L+\mu}+\gamma\big(\gamma-\frac{2}{L+\mu}\big) \mu^{2} \\
#     &=1+\gamma^{2} \mu^{2}-\frac{2\gamma\mu}{L+\mu}  (L + \mu) \\
#     &=(1-\gamma\mu)^{2}
#     \end{align*}
#   erhalten wir schließlich
#     \begin{equation*} 
#     \|x_{t+1} - x_\ast\|_2^2
#     \leq \max\big((1-\gamma L)^{2}, (1-\gamma\mu)^{2}\big)\ \|x_t - x_\ast\|_2^2
#     \end{equation*}
#     
# $\square$

# ## Zusammenfassung

# Ist $f=g+h$, $g$ differenzierbar, $L$-glatt und $\mu$-konvex,
#   $h$ konvex, $x_\ast = \mathrm{argmin}_x f(x)$, dann hat
#   Proximal-Gradient-Descent die selbe asymptotische Konvergenzgeschwindigkeit
#   wie im Fall dass $f$ differenzierbar, $L$-glatt und $\mu$-konvex ist.
# Die fehlende Glattheit von $h$ spielt, anders als bei Subgradient-Descent,
#   keine Rolle.
# 
#   
# Das Berechnen von 
# 
# $$
# \mathrm{prox}(x) =\underset{y}{\mathrm{argmin}} 
#       \big(
#       \frac{1}{2} \|y - x\|_2^2 + \gamma h(y)
#       \big)
# $$
# 
# ist in der Regel nicht trivial.
# Für einige spezielle $h$, die in der Praxis häufig auftreten, 
#   lässt sich $\mathrm{prox}(x)$ einfach bestimmen (siehe Übung).
#   
# Projected-Gradient-Descent kann als Spezialfall von Proximal-Gradient-Descent
#   interpretiert werden (siehe Übung).
