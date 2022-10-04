#!/usr/bin/env python
# coding: utf-8

# # Grundlagen der Optimierung

# ## Überblick

# In diesem Abschnitt betrachten wir verschiedene Methoden aus der Analysis
# zur Behandlung von Optimierungsproblemen.

# ## Grundlagen

# Wir betrachten eine Zielfunktion $f:\mathbb{R} ^{d}\rightarrow \mathbb{R}$. Minimiert man
# 
# - $f$ über ganz $\mathbb{R}^d$, so liegt ein 
#   *nicht restringiertes* Problem vor
# 
# - $f$ über $\emptyset \neq X\subset \mathbb{R} ^{d}$, so handelt es 
#   sich um ein *restringiertes* Problem
# 
# Ist $f(x) \ge c > -\infty$, so existiert $f_\ast = \inf_x f$, 
#   aber nicht notwendig ein $x_\ast$ mit $f(x_\ast)=\inf_x f = f_\ast$

# **Beispiel:** 
# 
# - für $f( x) =e^{-x^{2}}$, $x \in \mathbb{R}$, ist
#   \begin{equation*} 
#   \inf _{x\in \mathbb{R} }f\left( x\right) =0
#   \end{equation*}
#   aber es gibt kein $x_{\ast }\in \mathbb{R}$ mit
#   $f( x_{\ast }) =0$.

# Ist $f$ stetig, $X\subset \mathbb{R} ^{d}$ kompakt, dann hat das restringierte Problem (mindestens) eine Lösung, d.h.
# \begin{equation*} 
# \text{argmin}_{x\in X}f(x) \neq \emptyset 
# \end{equation*}
# 

# ## Nicht restringierte Probleme

# Im nicht restringierten Fall erhält man bei (ausreichend oft) differenzierbarer Zielfunktion $f$ notwendige und hinreichende Bedingungen für (lokale) Minimalstellen.
# 
# Ist $x_\ast \in \text{argmin}_{x\in \mathbb{R} ^{d}}f(x)$, dann gelten die notwendigen Bedingungen:
# 
# - ist $f\in C^{1}( \mathbb{R} ^{d})$ 
#     $\Rightarrow$ $f'( x_{\ast }) =0$ 
#     
# - ist $f\in C^{2}( \mathbb{R} ^{d})$ 
#     $\Rightarrow$ $f'( x_{\ast }) =0$ und $f''( x_{\ast })$ ist positiv semidefinit
#     
# Als hinreichende Bedingungen erhalten wir für $f\in C^{2}( \mathbb{R} ^{d})$:
# 
# - ist $f'(\tilde{x}) =0$ und $f''(\tilde{x})$ positiv definit
#   oder $f''(x)$ in einer offenen Umgebung $U$ von $\tilde{x}$ positiv semidefinit, 
#   dann gilt 
#   \begin{equation*} 
#   \tilde{x} \in \text{argmin}_{x\in U}f(x), 
#   \end{equation*}
#   d.h. $\tilde{x}$ ist lokales Minimum

# ## Restringierte Probleme, Lagrange-Funktion, KKT-Bedingungen

# Das Äquivalent zu den obigen Ergebnissen bei der restringierten Optimierung sind die 
# [*Karush-Kuhn-Tucker-Bedingungen*](https://de.wikipedia.org/wiki/Karush-Kuhn-Tucker-Bedingungen).
# 
# Wir betrachten das Optimierungsproblem
# \begin{equation*} 
# \min _{x\in X}f(x),
# \quad
# g(x) \leq 0,
# \quad
# h(x) =0,
# \end{equation*}
# mit $\emptyset\neq X \subset \mathbb{R} ^{d}$ und
# \begin{equation*} 
# f:X\rightarrow \mathbb{R} ,
# \quad
# g:X\rightarrow \mathbb{R} ^{m},
# \quad
# h:X\rightarrow \mathbb{R} ^{p}.
# \end{equation*}
# 
# Wir nehmen an, dass $f,g,h \in C^1(X)$ und definieren die zugehörige *Lagrange-Funktion* $L$ durch
# \begin{equation*} 
# L\left( x,\lambda ,\mu \right) =f\left( x\right) +\lambda ^{T}g\left( x\right) +\mu ^{T}h\left( x\right). 
# \end{equation*}
# 
# $x_\ast,\lambda_\ast,\mu_\ast$ heißt KKT-Punkt, falls
# \begin{equation*} 
# \begin{aligned}\partial _{x}L\left( x_{\ast },\lambda _{\ast },\mu _{\ast }\right) &=0,\\ 
# g\left( x_{\ast }\right) &\leq 0\\ 
# h\left( x_{\ast}\right) &=0,\\ 
# \lambda _{\ast } &\geq 0,\\ 
# \lambda _{\ast,i }g_i\left( x_{\ast }\right) &=0 \quad \forall i\end{aligned}
# \end{equation*}
# 
# Ist $x_\ast$ eine Lösung des Optimierungsproblems und erfüllt gewisse Regularitätsbedingungen (Constraint-Qualifications), dann existieren $\lambda_\ast \geq 0, \mu_\ast$, so dass $x_\ast,\lambda_\ast,\mu_\ast$ ein KKT-Punkt ist.
# Wir haben es hier also mit notwendige Bedingungen zu tun.

# Sind $X, f, g_i$ konvex und $h$ linear affin, dann vereinfacht sich das ganze noch:
# 
# - als Constraint-Qualification muss nur die 
#   *Slater-Bedingung* gelten, d.h.
#   \begin{equation*} 
#   \exists \tilde{x}\in X 
#   \quad\text{mit}\quad
#   g( \tilde{x}) <0,
#   \quad
#   h( \tilde{x})=0
#   \end{equation*}
#   
# - KKT ist auch hinreichend, d.h.$x_\ast$ ist lokales (und damit auch 
#   globales) Minimum, ohne weitere Annahmen

# **Beispiel:**
# 
# - wir betrachten das Problem
#   \begin{equation*} 
#   \min_{x\in \mathbb{R}^{2},\ x_{1}^{2}+x_{2}^{2}\leq 1,\ x_{1}=x_{2}}x_1
#   \end{equation*}
#   
# - mit $X=\mathbb{R}^{2}$,
#   \begin{equation*} 
#   f(x) = x_1, \quad g(x) = x_{1}^{2}+x_{2}^{2}-1, \quad h(x) = x_1 - x_2
#   \end{equation*}
#   erhalten wir die Standardform
#   \begin{equation*} 
#   \min _{x\in X}f(x),
#   \quad
#   g(x) \leq 0,
#   \quad
#   h(x) =0
#   \end{equation*}
#   
# - $X,f,g$ sind konvex, $h$ ist linear affin und die Slater-Bedingung ist ebenfalls erfüllt
#   
# - als Lagrange-Funktion erhalten wir
#   \begin{equation*} 
#   L(x,\lambda,\mu) = x_1 + \lambda(x_{1}^{2}+x_{2}^{2}-1) + \mu (x_1 - x_2)
#   \end{equation*}
#   und somit
#   \begin{align*}
#   \partial_{x}L(x,\lambda ,\mu) &=
#   \begin{pmatrix} 1 +2\lambda x_{1} +\mu \\ 2\lambda x_{2} -\mu \end{pmatrix} =0, \\
#   g\left( x\right) &=x_{1}^{2}+x_{2}^{2}-1\leq 0,\\ 
#   h\left( x\right) &=x_{1}-x_{2}=0,\\ 
#   \lambda g\left( x\right) &=\lambda \left( x_{1}^{2}+x_{2}^{2}-1\right) =0
#   \end{align*}
#   
# - subtrahiert man im Gradienten die zweite von der ersten Gleichung, so folgt
#   \begin{equation*} 
#   1+2\lambda \left( x_{1}-x_{2}\right) +2\mu =0
#   \end{equation*}
#   und wegen $x_{1}-x_{2}=0$ schließlich
#   \begin{equation*} 
#   \mu =-\dfrac{1}{2}
#   \end{equation*}
#   
# - eingesetzt in die zweite Komponente des Gradienten erhalten wir
#   \begin{equation*} 
#   \lambda x_{2}=\dfrac{\mu }{2}=-\dfrac{1}{4}
#   \end{equation*}
#   
# - aus $\lambda \geq 0$ folgt daraus $\lambda > 0$ und 
#     \begin{equation*} 
#     x_2 < 0
#     \end{equation*}
#     und wegen $\lambda \left( x_{1}^{2}+x_{2}^{2}-1\right) =0$ somit
#     \begin{equation*} 
#     x_{1}^{2}+x_{2}^{2} = 1
#     \end{equation*}
#     
# - außerdem soll noch die Gleichheitsbedingung
#   \begin{equation*} 
#   x_1 = x_2
#   \end{equation*}
#   gelten
#   
# - insgesamt erhalten wir damit
#   \begin{equation*} 
#   x_{2}=-\dfrac{1}{\sqrt{2}}
#   \end{equation*}
#   und somit ist
#   \begin{equation*} 
#   x_{\ast }=-\dfrac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}
#   \end{equation*}
#   wegen der Konvexität und der Slater-Bedingung die (eindeutige) Lösung des Optimierungsproblems

# Im nicht-konvexen Fall gelten für $f,g,h \in C^2$ folgende hinreichende Bedingungen:
# 
# - ist $x_\ast,\lambda_\ast,\mu_\ast$ ein KKT-Punkt und gilt
#   \begin{equation*} 
#   s^{T}\partial _{x}^{2}L\left( x_{\ast },\lambda _{\ast },\mu_\ast \right) s\geq 0
#   \end{equation*}
#   für alle $s\neq 0$ mit
#   \begin{equation*} 
#   \begin{pmatrix} 
#   \partial _{x}g_{i}\left( x_{\ast }\right) \\     
#   \partial _{x}h_{j}\left( x_{\ast }\ \right)
#   \end{pmatrix}^{T}
#   s=0
#   \end{equation*} 
#   mit $i$ s.d. $\lambda_{\ast,i}>0$,
#   dann ist $x_\ast$ ein lokales Minimum

# ## Dualität

# Wir betrachten das primale Problem
# \begin{equation*} 
# \min _{x\in X}f(x),
# \quad
# g(x) \leq 0,
# \quad
# h(x) =0,
# \end{equation*} 
# die zugehörige Lagrange-Funktion
# \begin{equation*} 
# L( x,\lambda ,\mu ) 
# = f(x) +\lambda ^{T}g(x) +\mu ^{T}h(x) 
# \end{equation*}
# und definieren damit die *duale Funktion*
# \begin{equation*} 
# q( \lambda ,\mu ) =\inf _{x\in X} L( x,\lambda ,\mu ). 
# \end{equation*}
# 
# Für $q$ untersuchen wir das duale Problem
# \begin{equation*} 
# \max_{\lambda \geq 0,\mu }q( \lambda ,\mu )
# \end{equation*}
# über dem *wesentlichen Zulässigkeitsbereich*
# \begin{equation*} 
# \mathrm{dom}_{q}=\big\{ ( \lambda ,\mu )\ \big|\  \lambda \geq 0,\ q( \lambda ,\mu ) >-\infty \big\}
# \end{equation*}

# Das duale Problem hat folgende Eigenschaften:
# 
#   - $\mathrm{dom}_{q}$ ist immer konvex
#   
#   - $-q$ ist immer konvex
#   
#   - ist 
#     \begin{equation*} 
#     R_{p}=\big\{ x \ \big| \ g( x) \leq 0,h( x) = 0\big\} 
#     \end{equation*} 
#     der zulässige Bereich des primalen Problems, dann gilt
#     \begin{equation*} 
#     \sup_{\mathrm{dom}_{q}} q\left( \lambda ,\mu \right) \leq \inf_{R_{p}} f( x)
#     \end{equation*}
#     
#     und somit gilt für alle Lösungen $\lambda_\ast, \mu_\ast$ von 
#     $\sup_{\mathrm{dom}_{q}} q\left( \lambda ,\mu \right)$ und $x_\ast$
#     von $\inf_{R_{p}} f( x)$
#     \begin{equation*} 
#     q(\lambda_\ast, \mu_\ast) \leq f(x_\ast).
#     \end{equation*}

# Die Differenz
# \begin{equation*} 
# f(x_\ast) - q(\lambda_\ast, \mu_\ast) \ge 0
# \end{equation*}
# bezeichnet man als *Dualitäts-Lücke*.
# Gilt
# \begin{equation*} 
# f(x_\ast) - q(\lambda_\ast, \mu_\ast) = 0
# \end{equation*}
# so spricht man von starker Dualität:
# 
# - starke Dualität gilt nicht immer
# 
# - sind $f,g$ konvex, $h$ linear affin und die Slater-Bedingung erfüllt, 
#   dann gilt starke Dualität
#   
# Praktischer Einsatz:
# 
# - ist nur der Wert $f_\ast = f(x_\ast)$ (und nicht $x_\ast$) interessant, 
#   dann kann man statt des primalen das duale Problem lösen, das immer konvex ist
#   
# - ist $\bar{x}$ eine Näherung von $x_\ast$ und gilt starke Dualität, so kann man mit Hilfe geeignet gewählter $\bar{\lambda},\bar{\mu}$ über
#   \begin{equation*} 
#   f(\bar{x}) - q(\bar{\lambda},\bar{\mu})
#   \end{equation*}
#   einen Fehlerindikator für $\bar{x}$ berechnen

# **Beispiel:**
# 
# - wir betrachten
#   \begin{equation*} 
#   \min _{x\in \mathbb{R} ^{d}} c^{T}x,
#   \quad Ax-b\leq 0,
#   \quad A\in \mathbb{R} ^{m\times d}
#   \end{equation*}
#   
# - als Lagrange-Funktion erhalten wir mit $\lambda \in \mathbb{R}^m$,
#   $\lambda \geq 0$
#   \begin{align*} 
#   L(x,\lambda) 
#   &= c^{T}x+\lambda ^{T}( Ax-b) \\
#   &=( c^{T}+\lambda ^{T}A) x - \lambda ^{T}b
#   \end{align*}
#   bzw. mit $y = c + A^T \lambda$
#   \begin{align*} 
#   L(x,\lambda) 
#   &=y^T x - \lambda ^{T}b  
#   \end{align*}
#   
# - jetzt berechnen wir die duale Funktion
#   \begin{equation*} 
#   q(\lambda) =\inf _{x\in \mathbb{R}^d} L( x, \lambda) 
#   \end{equation*}
# 
# - $L$ ist linear affin in $x$ mit
#     \begin{equation*} 
#     \partial_x L(x,\lambda) = y^T
#     \end{equation*}
#     
# - ist $y \neq 0$ so erhalten wir mit $x = \nu y$
#     \begin{equation*} 
#       L(x,\lambda) 
#       = \nu \|y\|_2^2 - \lambda ^{T}b   
#       \xrightarrow{\nu\to-\infty}-\infty
#     \end{equation*}
#     
# - für $y=0$ (also $\partial_x L(x,\lambda) = 0$) 
#   gilt
#     \begin{equation*} 
#       L(x,\lambda) 
#       = - \lambda ^{T}b
#       = \inf _{x\in \mathbb{R}^d} L( x, \lambda) 
#     \end{equation*}
# 
# - somit ist
#   \begin{equation*} 
#   q(\lambda) 
#   =\begin{cases}
#   -\lambda ^{T}b & \text{für }c + A^T \lambda=0\\ 
#   -\infty & \text{sonst}\end{cases}
#   \end{equation*}
#   und das duale Problem hat die Form
#   \begin{equation*} 
#   \max_{\lambda \in \mathbb{R} ^{m}}(-\lambda ^{T}b),
#   \quad \lambda \geq 0,
#   \quad A^{T}\lambda =-c
#   \end{equation*}

# **Beispiel:** 
# 
# - wir leiten jetzt die duale Form des Optimierungsproblems 
#   bei Support-Vector Classifiern her, die wir oben im Zusammenhang mit 
#   dem Kernel-Trick benutzt haben
# 
# - das primale Problem lautet
#     \begin{equation*} 
#     \min_{v\neq 0, v_0,\xi}
#     \Big(
#     \frac{1}{2}\|v\|_2^2
#     + C \sum_{i=1}^n \xi_i
#     \Big)
#     \end{equation*}
#     mit
#     \begin{equation*} 
#     y_i(v^T x_i + v_0)  \geq 1 - \xi_i, 
#     \quad
#     \xi_i\geq 0,
#     \quad i = 1,\ldots,n.
#     \end{equation*}
#     
# - setzen wir
#     \begin{align*} 
#     w 
#     &= (v,v_0,\xi) \in X = \mathbb{R}^{m + 1 + n} \\
#     f(w) 
#     &= \frac{1}{2}\|v\|_2^2 + C \sum_{i=1}^n \xi_i \\
#     g_i(w) 
#     &= -\Big(y_i \big(\sum_{j=1}^m v_j x_{ij} + v_0 \big) - 1 + \xi_i \Big),   
#     \quad i=1,\ldots,n\\
#     g_i(w) 
#     &= - \xi_i, 
#     \quad i=n+1,\ldots,2n
#     \end{align*}
#   so erhalten wir die Standardform
#     \begin{align*} 
#    \min _{w\in \mathbb{R}^{m + 1 + n}}f(w),
#     \quad
#     g(w) \leq 0
#     \end{align*}      

# - mit $\lambda = (\alpha, \beta)$, $\alpha,\beta \geq 0$, folgt für 
#   die Lagrange-Funktion
#     \begin{align*} 
#     L(w,\lambda) =&
#     L(v, v_0, \xi, \alpha, \beta) \\
#     = &
#     \frac{1}{2}\|v\|_2^2
#     + C \sum_{i=1}^n \xi_i
#     \\
#     &- \sum_{i=1}^n \alpha_i  \Big(y_i \big(\sum_{j=1}^m v_j x_{ij} + v_0 \big) - 1 + \xi_i \Big)\\
#     &- \sum_{i=1}^n\beta_i \xi_i\\
#     = &
#     \frac{1}{2}\|v\|_2^2
#     - \sum_{i=1}^n \alpha_i y_i \sum_{j=1}^m v_j x_{ij} 
#     \\
#     &- v_0\sum_{i=1}^n \alpha_i  y_i 
#     \\
#     &+ \sum_{i=1}^n (C-\alpha_i - \beta_i) \xi_i 
#     \\
#     &+ \sum_{i=1}^n \alpha_i
#     \end{align*}
#   bzw. für die duale Funktion
#     \begin{align*} 
#     q(\alpha,\beta) 
#     = \inf_{(v, v_0, \xi)\in \mathbb{R}^{m + 1 + n}} L(v, v_0, \xi, \alpha, \beta) 
#     \end{align*}  

# - $L$ ist quadratisch in $v$ und linear affin in $v_0$ 
#   und $\xi$ (und somit konvex und differenzierbar)
#   
# - wie im vorherigen Beispiel folgt, dass $q(\alpha,\beta)$
#   nur dann größer $-\infty$ sein kann, wenn
#     \begin{align*}
#     0 = \partial_{v_0} L(v, v_0, \xi, \alpha, \beta) 
#     &= - \sum_{i=1}^n \alpha_i y_i, 
#     \\
#     0 = \partial_{\xi_i} L(v, v_0, \xi, \alpha, \beta) 
#     &= 
#     C - \alpha_i - \beta_i,
#     \quad i = 1,\ldots,n,
#     \end{align*}
#   gilt, insgesamt also
#   \begin{align*}
#   \alpha_i,\beta_i\geq 0,
#   \quad
#   \sum_{i=1}^n \alpha_i y_i = 0, 
#   \quad
#   \alpha_i + \beta_i = C
#   \end{align*}

# - unter diesen Einschränkungen vereinfacht sich $L$ zu
#     \begin{align*} 
#     L(v, v_0, \xi, \alpha, \beta)
#     =
#     \frac{1}{2}\|v\|_2^2
#     - \sum_{i=1}^n \alpha_i y_i \sum_{j=1}^m v_j x_{ij} 
#     &+ \sum_{i=1}^n \alpha_i,
#     \end{align*}
#   ist also insbesondere unabhängig von $v_0, \xi, \beta$
#   und wird bei gegebenem $\alpha$ minimal an $\hat{v}$ mit
#     \begin{align*}
#     0 = \partial_{v_k} L(\hat{v}, v_0, \xi, \alpha, \beta) 
#     &= 
#     \hat{v}_k
#     - \sum_{i=1}^n \alpha_i y_i x_{ik},
#     \quad k = 1,\ldots,m,
#     \end{align*}
#   also
#   \begin{align*}
#   \hat{v}_k &= \sum_{i=1}^n \alpha_i y_i x_{ik}.\\
#   \end{align*}

# - eingesetzt in $L$ erhalten wir damit
#     \begin{align*} 
#     q(\alpha,\beta) 
#     & =   L(\hat{v}, v_0, \xi, \alpha, \beta)
#     \\
#     & = 
#     \frac{1}{2}\|\hat{v}\|_2^2
#     - \sum_{i=1}^n \alpha_i y_i \sum_{j=1}^m \hat{v}_j x_{ij} 
#     + \sum_{i=1}^n \alpha_i
#     \\
#     & =
#     \frac{1}{2}\sum_{j=1}^m \hat{v}_j^2
#     - \sum_{j=1}^m \hat{v}_j\sum_{i=1}^n \alpha_i y_i  x_{ij} 
#       +  \sum_{i=1}^n \alpha_i 
#       \\
#     & =
#     \frac{1}{2}\sum_{j=1}^m \hat{v}_j^2
#      - \sum_{j=1}^m \hat{v}_j^2
#       +  \sum_{i=1}^n \alpha_i 
#       \\
#     & =
#     -\frac{1}{2}\sum_{j=1}^m \hat{v}_j^2
#       +  \sum_{i=1}^n \alpha_i 
#       \\
#     & =
#     -\frac{1}{2}
#     \sum_{j=1}^m \big(\sum_{i=1}^n \alpha_i y_i  x_{ij} \big)^2
#       +  \sum_{i=1}^n \alpha_i \\
#     & =
#     -\frac{1}{2}
#     \sum_{i=1}^n \sum_{k=1}^n 
#     \alpha_i y_i 
#     \big( \sum_{j=1}^m x_{ij} x_{kj}\big)
#     y_k\alpha_k 
#      +  \sum_{i=1}^n \alpha_i 
#     \end{align*}  
#   bzw.
#     \begin{equation*} 
#     q(\alpha,\beta) =
#     -\frac{1}{2}\alpha^T Q \alpha + e^T \alpha 
#     \end{equation*}
#     mit
#     \begin{align*} 
#     Q = \big( y_i x_i^T x_k y_k \big)_{i,k = 1,\ldots,n},
#     \quad
#     e = (1,\ldots,1)^T
#     \end{align*}

# - insgesamt erhalten wir für die duale Funktion $q$ für
#   $\alpha,\beta\geq 0$
#   \begin{align*}
#   q(\alpha,\beta) 
#   =
#   \begin{cases}
#       -\frac{1}{2}\alpha^T Q \alpha + e^T \alpha 
#       & \text{für} \quad y^T\alpha = 0, \quad \alpha + \beta = Ce \\
#       - \infty & \text{sonst}
#   \end{cases}
#   \end{align*}

# - als duales Problem haben wir damit
#   \begin{align*}
#   \max_{\alpha\geq 0, \beta\geq 0}q(\alpha,\beta)
#   = \min_{\alpha\geq 0} 
#     \big(\frac{1}{2}\alpha^T Q \alpha - e^T \alpha \big)
#   \end{align*}
#   unter den Nebenbedingungen
#   \begin{align*}
#   y^T \alpha = 0,
#   \quad \alpha_i + \beta_i = C \quad \forall i
#   \end{align*}
# 
# - da $\beta_i \geq 0$ nur in die Nebenbedingung eingeht,
#   kann diese auf $0\leq \alpha_i \leq C$ geändert werden
#   
# - somit erhalten wir die finale Form des dualen Problems,
#   wie wir sie bei den Support-Vector Classifiern bereits 
#   benutzt haben: 
#     \begin{equation*} 
#     \min_{\alpha \geq 0}
#     \Big(
#     \frac{1}{2}\alpha^T Q \alpha - e^T \alpha 
#     \Big)
#     \end{equation*}
#     mit
#     \begin{equation*} 
#     Q = \big( y_i x_i^T x_jy_j \big)_{i,j = 1,\ldots,n},
#     \quad
#     y^T \alpha = 0,
#     \quad
#     0\leq \alpha \leq Ce.
#     \end{equation*}

# ## Zusammenfassung

# Für nicht-restringierte bzw, restringierte Optimierungsprobleme haben wir die Werkzeuge
# aus der Analysis betrachtet, u.a. Lagrange-Funktion, KKT-Bedingungen und duale Probleme.
