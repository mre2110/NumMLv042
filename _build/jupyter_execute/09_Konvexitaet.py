#!/usr/bin/env python
# coding: utf-8

# # Konvexität

# ## Überblick

# - konvexe Mengen und Funktionen spielen eine wichtige Rolle in der Optimierung
# 
# - konvexe Optimierungsprobleme lassen sich sowohl theoretisch als
#   auch algorithmisch relativ gut behandeln
#   
# - in diesem Kapitel werden einige elementare Ergebnisse aus diesem Bereich
#   hergeleitet

# ## Konvexe Mengen und konvexe Funktionen

# **Definition:** 
# $X\subset\mathbb{R}^d$ heißt konvex, falls
#   \begin{equation*} 
#   (1-\lambda ) x+\lambda y\in X 
#   \quad 
#   \forall x,y\in X
#   \quad 
#   \forall \lambda \in [ 0,1]
#   \end{equation*}
#   gilt
#   
#   
# Sind $X_i$, $i\in I$ konvex, dann ist auch $\bigcap_{i \in I}X_i$ konvex

# Um Konvexität von Funktionen zu untersuchen, führen wir einige Begriffe ein.
# Wir betrachten $f:\mathrm{dom}(f) \rightarrow \mathbb{R}$, wobei
# $\mathrm{dom}(f) \subset \mathbb{R} ^{d}$ der Definitionsbereich von $f$ ist.

# **Definition:**  
# 
# - $\mathrm{graph}( f) 
#   =\big\{ \big( x,f( x) \big) \ \big| \ x\in \mathrm{dom}( f) \big\} \subset \mathbb{R} ^{d+1}$
#   
# - $\mathrm{epi}( f) 
#   =\big\{ ( x,y) \ \big| \ y \geq f( x), \ x\in \mathrm{dom}( f) \big\}$, 
#   ist der *Epigraph* von $f$

# **Definition**:
# 
# - $f:\mathrm{dom}(f) \rightarrow \mathbb{R}$ ist *konvex*, falls $\mathrm{dom}(f)$ konvex ist und
#     \begin{equation*} 
#     f\big( ( 1-\lambda) x+\lambda y\big) \leq ( 1-\lambda ) f( x) +\lambda f( y) \quad \forall x,y\in \mathrm{dom}(f) \quad \forall \lambda \in [ 0,1] 
#     \end{equation*}
# 
# - $f:\mathrm{dom}(f) \rightarrow \mathbb{R}$ ist *strikt konvex*, falls $\mathrm{dom}(f)$ konvex ist und
#     \begin{equation*} 
#     f\big( ( 1-\lambda) x+\lambda y\big) < ( 1-\lambda ) f( x) +\lambda f( y) \quad \forall x\neq y\in \mathrm{dom}(f) \quad \forall \lambda \in ( 0,1) 
#     \end{equation*}
# 
# - $f$ ist $\mu$*-konvex*, $\mu>0$, falls
#     \begin{equation*} 
#     f(x) - \frac{\mu}{2} \|x\|_2^2
#     \end{equation*}
#     konvex ist

# **Beispiel:** 
# 
# - alle Normen sind konvex (Dreiecksungleichung)
# 
# - Least Square Loss
# 
# - $f(x) = x^TQx + b^Tx + c$ mit $Q$ symmetrisch positiv semidefinit, 
#   also insbesondere auch für $Q=0$ und somit
#   alle linear affinen Funktionen $f(x) = b^Tx + c$
# 
# - $f(x) = x^TQx + b^Tx + c$ mit $Q$ symmetrisch positiv definit 
#   ist $\mu$ konvex für $0<\mu \leq \lambda_{\min}(Q)$

# **Rechenregeln:** 
# 
# - sind $f,g$ konvex dann gilt dies auch für
# 
#   - $\alpha f$ mit $\alpha \geq 0$
# 
#   - $f+g$
# 
#   - $f(Ax+b) \ \forall A,b$
# 
#   - $h(x) = \max\big(f(x),g(x)\big)$
# 
# - mit $f(x,y)$ ist auch $g(x)=\min_y f(x,y)$ konvex
# 
# - für $f,g$ konvex ist $f\circ g$ i.d.R. nicht konvex
# 
# - $f:\mathrm{dom}(f) \rightarrow \mathbb{R}$ ist konvex genau dann, wenn $\mathrm{epi}(f)$ konvex ist (Beweis: siehe Übung)
# 
# - Jensen-Ungleichung: $f:\mathrm{dom}(f) \rightarrow \mathbb{R}$ konvex, $x_i\in\mathrm{dom}(f)$, $i=1,\ldots,n$, dann gilt
#     \begin{equation*} 
#     f\Big(\sum_{i=1}^n \lambda_i x_i\Big) \le \sum_{i=1}^n \lambda_i f(x_i)
#     \quad \forall \lambda_i \geq 0 \quad \text{mit} \quad \sum_{i=1}^n \lambda_i =1
#     \end{equation*} 
#     (Beweis: siehe Übung)

# ## Eigenschaften konvexer Funktionen

# **Satz:** Ist $f$ konvex, dann sind lokale Minima immer globale Minima
# 
# 
# **Beweis:**
# 
# - sei $x$ ein lokales Minimum, d.h.
#   \begin{equation*} 
#   f( x) \leq f( y) \quad 
#   \forall y \quad \text{mit} \quad \| y-x\| < r,
#   \end{equation*}
#   $r$ hinreichend klein
# 
# - Annahme: es existiert $\bar{x}\in\mathrm{dom}(f)$ mit $f(\bar{x})<f(x)$
# 
# - für $\lambda>0$ sei
#   \begin{equation*} 
#   y=x+\lambda ( \bar{x}-x) =( 1-\lambda ) x+\lambda \bar{x}
#   \end{equation*}
# - damit ist
#   \begin{equation*} 
#   \| y-x\| = \lambda \| x-\bar{x}\|
#   \end{equation*}
#   
# - da $\mathrm{dom}(f)$ konvex ist, ist $y\in \mathrm{dom}(f)$
#   für $\lambda\in[0,1]$
#   und somit 
#   \begin{align*}
#   f( y) 
#   &\leq ( 1-\lambda ) f( x) +\lambda f( \bar{x}) \\ 
#   &= f( x) +\lambda \big( \underbrace{f( \bar{x}) -f( x)}_{<0} \big) \\ 
#   &< f( x)
#   \end{align*}
#   für alle $\lambda\in(0,1]$
# 
# - wählt man jetzt $\lambda\in(0,1]$ klein genug, so dass $\| y-x\| < r$ gilt,
#   so erhalten wir einen 
#   Widerspruch dazu, dass $x$ ein lokales Minimum ist
# 
# $\square$

# **Bemerkung:** Konvexe Funktionen müssen keine lokalen oder globalen Minima besitzen, z.B. $f(x)=e^x$

# **Satz:** Ist $f$ strikt konvex, dann gibt es höchstens ein Minimum
# 
# 
# **Beweis:**
# 
# - es seien $x\neq y$ Minima, d.h.
#     \begin{equation*} 
#     f(x) = f(y) = f_\ast = \inf_x f(x)
#     \end{equation*}
# 
# - mit der Konvexität von $f$ folgt
#     \begin{equation*} 
#     z = \frac{1}{2}x+\frac{1}{2}y \in \mathrm{dom}(f)
#     \end{equation*}
#     und
#     \begin{equation*} 
#     f(z)=
#     f\Big( \frac{1}{2}x+\frac{1}{2}y\Big) 
#     <\frac{1}{2}f( x) +\frac{1}{2}f( y) 
#     =f_\ast 
#     \end{equation*}
#     was ein Widerspruch zur Minimalität von $f_\ast$ ist
# 
# $\square$

# Für die Glattheit konvexer Funktionen liefert das folgende Lemma einen Hinweis.
# 
# 
# **Lemma:** Es sei $f$ konvex, $\bar{B}_\delta(x) \subset \mathrm{dom}(f)$
# und $f$ beschränkt auf $\bar{B}_\delta(x)$. Dann ist $f$ Lipschitz-stetig auf $\bar{B}_\delta(x)$.
# 
# 
# **Beweis:**
# 
# - mit $y \in \bar{B}_\delta(x)$, $x\neq y$, ist $0 < \|x-y\| \leq \delta$
# 
# - wir setzen nun
#   \begin{align*} 
#   z
#   &=x+\underbrace{\frac{\delta }{\| x-y\|}}_{\alpha}( x-y) 
#   \\
#   &=x+\alpha ( x-y) 
#   \\
#   &=( 1+\alpha ) x-\alpha y
#   \end{align*}
#   und erhalten
#   \begin{equation*} 
#   \|z -x\|=\frac{\delta}{\| x-y\|}\cdot \| y-x\|=\delta,
#   \end{equation*}
#   also $z \in \partial\bar{B}_\delta(x)$, bzw.
#   \begin{align*} 
#   x
#   &=\frac{1}{1+\alpha }( z+\alpha y) 
#   \\
#   &=\frac{1}{1+\alpha }z+\frac{\alpha }{1+\alpha }y
#   \\
#   &=\big(\underbrace{ 1-\frac{\alpha }{1+\alpha }}_{1-\beta}\big) z
#   +\underbrace{\frac{\alpha }{1+\alpha }}_{\beta}y
#   \end{align*}
#   
# - wegen $\alpha>0$ gilt $\beta\in[0,1]$ und aus der Konvexität von $f$ folgt
#   \begin{align*} 
#   f(x) 
#   &\leq (1-\beta)f(z) + \beta f(y)
#   \\
#   &= \frac{1}{1+\alpha } f(z) + \frac{\alpha }{1+\alpha }f(y)
#   \end{align*}
#   und somit
#   \begin{align*} 
#   f(x) - f(y)
#   &\leq \frac{1}{1+\alpha } f(z) + \frac{\alpha }{1+\alpha }f(y)  - f(y)
#   \\
#   &=  \frac{1}{1+\alpha } \big(f(z)-f(y)\big)
#   \end{align*}
# 
# - $f$ ist beschränkt auf  $\bar{B}_\delta(x)$, d.h.
#     \begin{equation*} 
#     |f(u)| \leq M \quad \forall u\in \bar{B}_\delta(x)
#     \end{equation*}
# - wegen $z,y \in \bar{B}_\delta(x)$ erhalten wir
#   \begin{align*} 
#   f(x) - f(y)
#   &\leq \frac{2M}{1+\alpha} 
#   \\
#   &= \frac{2M}{1+\frac{\delta}{\| x-y\|}} 
#   \\
#   &= \frac{2M \| x-y\|}{\delta + \| x-y\|} 
#   \\
#   &\leq \frac{2M }{\delta}\| x-y\|
#   \end{align*}
# 
# - durch vertauschen von $x$ und $y$ folgt schließlich
#   \begin{equation*} 
#   |f(x) - f(y)| \leq \frac{2M }{\delta}\| x-y\|
#   \end{equation*}
# 
# $\square$

# Ist $f$ also konvex und beschränkt, dann ist es immer lokal Lipschitz-stetig.
# 
# Ist $X$ ein endlichdimensionaler normierter Raum, so kann man zeigen, dass jede konvexe Funktion $f:X \supset \mathrm{dom}(f) \rightarrow \mathbb{R}$ auf jeder Kugel $\bar{B}_\delta(x) \subset \mathrm{dom}(f)$ beschränkt ist, so dass wir in diesem Fall auf die zusätzliche Voraussetzung der Beschränktheit verzichten können

# Wie sieht es mit der Differenzierbarkeit konvexer Funktionen aus?
# 
# 
# **Lemma:** $f:\mathbb{R}^d \supset \mathrm{dom}(f) \rightarrow \mathbb{R}$ konvex,
# $\mathrm{dom}(f)$ konvex, offen, ist fast überall diffenrenzierbar, d.h.
# \begin{equation*} 
# \forall x\in \mathrm{dom}( f)
# \quad \forall \varepsilon >0
# \quad \exists \tilde{x}\in \mathrm{dom}( f)
# \quad\text{mit}
# \left\| \tilde{x}-x\right\| <\varepsilon
# \end{equation*}
# und $f$ ist differenzierbar in $\tilde{x}$
# 
# 
# Es gibt stetige Funktionen, die nirgends differenzierbar sind.
# Die Konvexität sorgt also für ein gewisses Maß an Regularität

# ## Differenzierbare konvexe Funktionen

# Bei differenzierbarem $f$ besteht ein enger Zusammenhang zwischen der Konvexität und den Eigenschaften der Ableitungen

# **Lemma:**
# $f:\mathbb{R}^d \supset \mathrm{dom}(f) \rightarrow \mathbb{R}$ sei differenzierbar.
# $f$ ist konvex genau dann wenn $\mathrm{dom}(f)$ konvex ist und
# \begin{equation*} 
# f(y) \geq f(x) + f'(x)(y-x) \quad \forall x,y \in \mathrm{dom}(f)
# \end{equation*}

# **Beweis:**
# 
# "$\Rightarrow$"
# 
# - ist $f$ konvex, so ist $\mathrm{dom}(f)$ konvex und
#   \begin{equation*} 
#   f\big((1-t) x+t y\big) \leq (1-t) f(x) + t f(y) \quad \forall t     
#   \in[0,1]
#   \end{equation*}
# 
# - für $t \in(0,1)$ gilt dann
#   \begin{equation*} 
#   \begin{aligned}
#   f\big(x+t(y-x)\big) 
#   & =f\big((1-t) x+t y\big) \\
#   & \leq (1-t) f(x)+t f(y)\\
#   &=f(x)+t\big(f(y)-f(x)\big)
#   \end{aligned}
#   \end{equation*}
#   und somit
#   \begin{equation*} 
#   f(y) \geq f(x) +\frac{f\big(x+t(y-x)\big)-f(x)}{t} \quad \forall t      \in(0,1)
#   \end{equation*}
# 
# - durch Grenzübergang $t\downarrow 0$ erhält man
#   \begin{equation*} 
#   f(y) \geq f(x) + f'(x)(y-x)
#   \end{equation*}
# 
# "$\Leftarrow$"
# 
# - es gilt $\mathrm{dom}(f)$ konvex und
#   \begin{equation*} 
#   f(y) \geq f(x) +f'(x) (y-x) \quad \forall x,y\in \mathrm{dom}(f)
#   \end{equation*}
# 
# - für $t\in \left[ 0,1\right]$ setzen wir
#   \begin{equation*} 
#   z= (1-t) x+ty
#   \end{equation*}
# 
# - dann ist $z\in \mathrm{dom}(f)$ und
#   \begin{equation*} 
#   \begin{aligned}
#   f(x) 
#   &\geq f(z) +f'(z)  (x-z) 
#   \\ 
#   f(y) 
#   &\geq f(z) +f'(z) (y-z)
#   \end{aligned}
#   \end{equation*}
# 
# - multiplizieren wir die erste Ungleichung mit $1-t$ und die zweite mit $t$ und addieren die Ergebnisse, dann erhalten wir
#   \begin{align*}
#   (1-t) f(x) + t f(y) 
#   &\geq (1-t)\big(f(z) +f'(z)  (x-z) \big) 
#   \\
#   &\quad + t \big(f(z) +f'(z) (y-
#   z)\big)\\
#   &= f(z) +f'(z) \big( \underbrace{( 1-t) x+ty}_z -z\big)\\
#   &= f(z) \\
#   &= f\big(( 1-t) x+ty \big)
#   \end{align*}
# 
# $\square$

# **Bemerkung:** 
# $f(y) \geq f(x) +f'(x) (y-x) =: t_{x}(y)$, d.h. der Graph von $f$ liegt immer oberhalb der Tangenten $t_{x}$

# In[1]:


import sympy as sy
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

fsize = 12

x = sy.symbols('x')
f  = sy.Lambda(x, x*x/2 + 1/2)
f1 = sy.Lambda(x, f(x).diff(x))

x0 = -1

tx = sy.Lambda(x, f(x0) + f1(x0)*(x-x0) )

tx = sy.lambdify(x, tx(x))
f  = sy.lambdify(x, f(x))

x = np.linspace(-2,1)
xmin = x.min()
xmax = x.max()
plt.plot(x, f(x) , 'g', label = '$f$')
plt.plot(x, tx(x), 'c', label = '$t_{x}$')
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


# Alternativ zeigt man
# 
# 
# **Lemma:** 
# $f:\mathbb{R}^d \supset \mathrm{dom}(f) \rightarrow \mathbb{R}$ sei differenzierbar.
# $f$ ist konvex genau dann wenn $\mathrm{dom}(f)$ konvex ist und
# \begin{equation*} 
# \big(f'(y) - f'(x)\big)(y-x) \geq 0 \quad \forall x,y \in \mathrm{dom}(f),
# \end{equation*}
# 
# 
# **Beweis:**
#   
# "$\Rightarrow$"
#   
# - ist $f$ konvex und differenzierbar, dann gilt
#     \begin{equation*} 
#     f(y) \geq f(x) +f'(x) (y-x)
#     \end{equation*}
#     bzw.
#     \begin{equation*} 
#     f(x) \geq f(y) +f'(y) (x-y)
#     \end{equation*}
#     
# - durch Addition der Ungleichunge erhält man
#     \begin{equation*} 
#     f(y) + f(x) \geq f(x) + f(y) + \big(f'(x) - f'(y)\big) (y-x)
#     \end{equation*}
#     und somit
#     \begin{equation*} 
#     \big(f'(y) - f'(x)\big)(y-x) \geq 0
#     \end{equation*}
#     
# "$\Leftarrow$"
#   
# - es gelte jetzt
#     \begin{equation*} 
#     \big(f'(y) - f'(x)\big)(y-x) \geq 0 \quad \forall x,y \in \mathrm{dom}(f)
#     \end{equation*}
#   
# - da $\mathrm{dom}(f)$ konvex ist, ist
#     für $s,\lambda \in [0,1]$ die Funktion
#     \begin{equation*} 
#     g(s) = f\big(x + s\lambda (y-x) \big)
#     \end{equation*}
#     wohldefiniert
#     
# - es ist
#     \begin{equation*} 
#     g(0) = f(x), \quad g(1) = f\big(x + \lambda (y-x) \big)
#     \end{equation*}
#     und mit $f$ ist auch $g$ differenzierbar mit
#     \begin{equation*} 
#     g'(s) = \lambda f'\big(x + s\lambda (y-x) \big)(y-x)
#     \end{equation*}
#    
# - damit folgt
#     \begin{align*}
#     f\big(x + \lambda (y-x) \big) - f(x)
#     & = g(1) - g(0) \\
#     & = \int_0^1 g'(\sigma) \:d\sigma \\
#     & = \int_0^1 \lambda f'\big(x + \sigma\lambda (y-x) \big)(y-x)\:d\sigma
#     \end{align*}
# 
# - aus der Voraussetzung folgt
#     \begin{align*}
#     0 
#     &\leq \big(  f'\big(x + \sigma (y-x) \big) 
#                - f'\big(x + \sigma\lambda (y-x) \big) \big)
#               \big(x + \sigma (y-x) - \big(x + \sigma\lambda (y-x) \big)\big)  \\
#     & = \sigma (1-\lambda)  \big(  f'\big(x + \sigma (y-x) \big) 
#                - f'\big(x + \sigma\lambda (y-x) \big) \big)
#                (y-x)      
#     \end{align*}
# 
# - für $\sigma,\lambda \in (0,1)$ ist $\sigma (1-\lambda) > 0$ und damit
#     \begin{equation*} 
#     f'\big(x + \sigma\lambda (y-x) \big)(y-x)
#     \leq
#     f'\big(x + \sigma (y-x) \big)(y-x)
#     \end{equation*}
#   
# - benutzen wir diese Abschätzung im Integral von oben, so erhalten wir
#     wegen $\lambda\in(0,1)$
#     \begin{align*}
#     f\big(x + \lambda (y-x) \big) - f(x)
#     & = \lambda \int_0^1 f'\big(x + \sigma\lambda (y-x) \big)(y-x)\:d\sigma \\
#     & \leq \lambda \int_0^1 f'\big(x + \sigma (y-x) \big)(y-x)\:d\sigma \\
#     & = \lambda \big(f(y) - f(x)\big)
#     \end{align*}
#     und damit
#     \begin{equation*} 
#     f\big( (1-\lambda)x + \lambda y \big) \leq (1-\lambda)f(x) + \lambda f(y)
#     \end{equation*}
#     
# - für $\lambda = 0$ bzw. $\lambda = 1$ gilt die Ungleichung trivialerweise
#     
# $\square$

# **Bemerkung:**
# $\big(f'(y) - f'(x)\big)(y-x) \geq 0$ bedeutet, das $f'$ monoton wachsend ist

# Oben haben wir gesehen, dass bei konvexem $f$ lokale Minima *immer* globale Minima sind.
# 
# 
# **Lemma:** $f:\mathbb{R}^d \supset \mathrm{dom}(f) \rightarrow \mathbb{R}$ konvex, differenzierbar, $\mathrm{dom}(f)$ offen. $x\in\mathrm{dom}(f)$ ist globales Minimum von $f$ genau dann wenn $f'(x)=0$
# 
# 
# **Beweis:**
# 
# "$\Rightarrow$" 
#   
# - $\mathrm{dom}(f)$ ist offen, also ist $x$ auch ein 
#   lokales Minimum und somit gilt $f'(x)=0$
#   
# "$\Leftarrow$" 
#   
# - $f(y) \geq f(x) +f'(x) (y-x) = f(x)\quad\forall y\in \mathrm{dom}(f)$
#   
# $\square$

# Jetzt betrachten wir zweite Ableitungen von $f$.
# 
# 
# **Lemma:** $f:\mathbb{R}^d \supset \mathrm{dom}(f) \rightarrow \mathbb{R}$ sei $C^2$, $\mathrm{dom}(f)$ offen.
# $f$ ist konvex genau dann wenn $\mathrm{dom}(f)$ konvex ist und
#   \begin{equation*} 
#   f''(x) \succeq 0 \quad \forall x \in \mathrm{dom}(f)
#   \end{equation*}
#   gilt ($f''(x) \succeq 0$ bedeutet $f''(x)$ ist positiv semidefinit)

# **Beweis:**
# 
# "$\Rightarrow$"
#   
# - da $\mathrm{dom}(f)$ offen ist gilt für hinreichend kleines $t\geq 0$
#     \begin{equation*} 
#     x + tv \in \mathrm{dom}(f) \quad \forall v \in \bar{B}_1(0)
#     \end{equation*}
#     
# - da $f$ konvex und damit $f'$ monoton wachsend ist erhält man für beliebiges  $v \in \bar{B}_1(0)$
#   \begin{align*}
#   v^T f''(x) v
#   &  = \lim_{t\downarrow 0} \frac{\big(f'(x + tv) - f'(x)\big)v}{t}\\
#   &  = \lim_{t\downarrow 0} \frac{\big(f'(x + tv) - f'(x)\big)tv}{t^2}\\
#   &  = \lim_{t\downarrow 0} \frac{\big(f'(x + tv) - f'(x)\big)
#        (x + tv - x)}{t^2}\\
#   &\geq 0
#   \end{align*}
#      
# "$\Leftarrow$"
# 
# - da $\mathrm{dom}(f)$ konvex ist, ist für $t\in[0,1]$ und $x,y \in \mathrm{dom}(f)$ die Funktion
#   \begin{equation*} 
#   g(t) = f'\big(x+ t (y-x)\big)(y-x)
#   \end{equation*}
#   wohldefiniert und auf $(0,1)$ differenzierbar mit Ableitung
#   \begin{equation*} 
#   g'(t) = (y-x)^T f''\big(x+ t (y-x)\big)(y-x)
#   \end{equation*}
#   
# - aus dem Mittelwertsatz folgt dann
#   \begin{align*}
#   \big(f'(y) - f'(x)\big)(y-x) 
#   &= g(1) - g(0)\\
#   &= g'(\tau) \\
#   &= (y-x)^T f''\big(x+ \tau (y-x)\big)(y-x)
#   \end{align*}
#   mit $\tau\in(0,1)$
#   
# - da $\mathrm{dom}(f)$ konvex ist, gilt für alle $\tau\in(0,1)$ 
#   und für alle $x,y\in \mathrm{dom}(f)$
#   \begin{equation*} 
#   \xi = x + \tau(y-x) \in \mathrm{dom}(f)
#   \end{equation*}
#   also
#   \begin{align*}
#   \big(f'(y) - f'(x)\big)(y-x) = (y-x)^Tf''(\xi)(y-x) \geq 0
#   \end{align*}
#   womit $f'$ monoton wachsend und damit $f$ konvex ist
#   
# $\square$

# **Bemerkung:**
# 
#   - ist $f''(x) \succ 0$ $\forall x \in \mathrm{dom}(f)$ (positiv definit), 
#     dann ist $f$ strikt konvex
# 
#   - die Umkehrung der letzten Aussage ist falsch, denn
#     $f(x)=x^4$ ist strikt konvex auf $\mathbb{R}$, aber $f''(0)=0$

# ## Restringierte konvexe Minimalprobleme

# **Problemstellung:**
#   
# - gegeben sei $f:\mathbb{R}^d \supset \mathrm{dom}(f) \rightarrow \mathbb{R}$ konvex, sowie
#     \begin{equation*} 
#     \emptyset\neq X \subset \mathrm{dom}(f) \quad \text{konvex}
#     \end{equation*}
#   
# - bestimme $x_\ast = \mathrm{argmin}_{x\in X} f(x)$

# **Satz:** $f,X$ wie in der Problemstellung, $\mathrm{dom}(f)$ offen, $f$ sei $C^1$. Dann gilt
# \begin{equation*} 
# x_\ast\in X \quad\text{ist Minimierer}
# \quad \Leftrightarrow \quad
# f'(x_\ast)(x-x_\ast)\geq 0 \quad\forall x\in X
# \end{equation*}

# **Beweis:**
# 
# "$\Rightarrow$"
#   
# - sei $x_\ast \in X$ Minimierer, und $x\in X$ mit
#     \begin{equation*} 
#     f'(x_\ast)(x-x_\ast) < 0
#     \end{equation*}  
# 
# - wir betrachten die Funktion
#   \begin{align*} 
#   g(t) &= f\big(x_\ast + t(x-x_\ast)\big),
#   \\
#   g'(t) &= f'\big(x_\ast + t(x-x_\ast)\big)(x-x_\ast)
#   \end{align*}
#   
# - wegen
#   \begin{align*} 
#   g(0) &= f(x_\ast),
#   \\
#   g'(0) &=  f'(x_\ast)(x-x_\ast) < 0
#   \end{align*}
#   existiert ein $t>0$ mit
#   \begin{equation*} 
#   f\big(x_\ast + t(x-x_\ast)\big)
#   = g(t)
#   < g(0)
#   = f(x_\ast)
#   \end{equation*}
# 
# - da $X$ konvex ist, ist $x_\ast + t(x-x_\ast) \in X$,
#   so dass $x_\ast$ kein Minimierer in $X$ sein kann
#   
# "$\Leftarrow$"
#   
# - $f$ ist konvex, weshalb
#   \begin{equation*} 
#   f(y) - f(x) 
#   \geq  f'(x)(y-x) 
#   \quad 
#   \forall x,y\in \mathrm{dom}(f)
#   \end{equation*}  
#   gilt
#    
# - mit $x = x_\ast$ folgt dann mit der Voraussetzung
#   \begin{equation*} 
#   f(y) - f(x_\ast) 
#   \geq  f'(x_\ast)(y-x_\ast) 
#   \geq 0 
#   \quad 
#   \forall y\in X \subset \mathrm{dom}(f)
#   \end{equation*}  
#   so dass $x_\ast \in X$ Minimierer ist
# 
# $\square$

# **Bemerkung:**
#   $f'(x_\ast)(x-x_\ast)$ sind alle "zulässigen" Richtungsableitungen, d.h. solche die "nicht aus $X$ hinaus gehen"

# Wann existiert nun ein Minimierer $x_\ast$?
# Ist $X$ kompakt, dann folgt aus der Stetigkeit von $f$ die Existenz.
# Ist $X$ nicht kompakt, dann kann $\inf_{x\in X}f(x) > -\infty$ sein, es muss aber kein $x_\ast$ existieren, wie. z.B. bei $f(x) = e^x$ auf $\mathbb{R}$.
# 
# Das ist in der Praxis oft nicht schlimm, man gibt sich für eine vorgegebene Toleranz $\varepsilon > 0$ mit einem $x_\varepsilon$ mit
# \begin{equation*} 
# f(x_\varepsilon) - f(x_\ast) \leq \varepsilon
# \end{equation*}
# zufrieden.
# 
# Für den Fall $X = \mathbb{R}^d$ erhalten wir hinreichende Existenzaussagen.

# **Definition:** $f:\mathbb{R}^d \rightarrow \mathbb{R}$, $\alpha\in\mathbb{R}$, dann heißt
# 
# \begin{equation*} 
# f^{\leq\alpha} =
# \{x \ |\ x\in\mathbb{R}^d ,\ f(x)\leq\alpha\}
# \end{equation*}
# 
# $\alpha$*-Sublevel-Menge* von $f$
# 
# 
# **Satz:** Ist $f:\mathbb{R}^d \rightarrow \mathbb{R}$ konvex und existiert ein 
# $\alpha\in\mathbb{R}$ so dass $f^{\leq\alpha}$ nichtleer und beschränkt ist, dann besitzt
# $f$ einen globalen Minimierer $x_\ast$
# 
# 
# **Beweis:**
# 
# - $f$ ist stetig und $f^{\leq\alpha}$ ist das Urbild von $(-\infty,\alpha]\subset\mathbb{R}$
#   
# - $(-\infty,\alpha]$ ist abgeschlossen in $\mathbb{R}$, also ist $f^{\leq\alpha}$
#   abgeschlossen in $\mathbb{R}^d$
#   
# - $f^{\leq\alpha}$ ist nach Voraussetzung auch beschränkt und somit kompakt
#   
# - da $f$ stetig ist, nimmt es auf $f^{\leq\alpha}$ sein Minimum an, d.h.
#     $\exists x_\ast \in f^{\leq\alpha}$ mit
#     \begin{equation*} 
#     x_\ast = \mathrm{argmin}_{x\in f^{\leq\alpha}} f(x),
#     \quad
#     f(x_\ast)\leq\alpha
#     \end{equation*}
#     
# - für $x\notin f^{\leq\alpha}$ gilt
#     \begin{equation*} 
#     f(x) \geq \alpha \geq f(x_\ast)
#     \end{equation*}
#     und somit ist $x_\ast$ globaler Minimierer
#     
# $\square$

# ## Zusammenfassung

# - bei konvexen Funktionen sind lokale auch immer globale Minima
# 
# - Konvexität kann bei differenzierbarem $f$ geprüft werden über
#   die Monotonie des Gradienten $f'$ bzw. die positive Semidefinitheit
#   der Hesse-Matrix $f''$
#   
# - ist $f$ konvex und differenzierbar, dann ist 
#   \begin{equation*} 
#   f'(x_\ast)=0
#   \end{equation*} 
#   hinreichend und notwendig
#   dafür, dass $x_\ast$ globales Minimum ist
#   
# - bei restringierter Optimierung konvexer, differenzierbarer Funktionen $f$
#   über konvexe Mengen $X$ ist
#   \begin{equation*} 
#   f'(x_\ast)(x-x_\ast)\geq 0 \quad\forall x\in X
#   \end{equation*} 
#   hinreichend und notwendig
#   dafür, dass $x_\ast$ globales Minimum ist
