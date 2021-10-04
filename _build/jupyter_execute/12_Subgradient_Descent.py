#!/usr/bin/env python
# coding: utf-8

# # Subgradient Descent

# ## Überblick

# Bisher haben wir immer $f\in C^1(\mathbb{R}^d)$ vorausgesetzt.
# In der Praxis trifft das nicht immer zu (Lasso, RELU-MLP).
# 
# Ist $f$ konvex, dann existiert in jedem Punkt der Subgradient.
# Wir benutzen bei Gradient Descent statt des 
# (eventuell nicht existierenden) Gradienten einen Subgradienten.
# 
# Die fehlende Glattheit von $f$ wird sich negativ auf die Konvergenzgeschwindigkeit auswirken.

# ## Grundlagen

# **Definition**:
# 
# - ist $f:\mathbb{R}^d \supset \mathrm{dom}(f) \rightarrow \mathbb{R}$, 
#     dann ist $g\in \mathbb{R}^d$ *Subgradient* von $f$ in $x\in \mathrm{dom}(f)$, falls
#     \begin{equation*} 
#     f(y) \geq f(x) + g^T(y-x) \quad \forall y\in\mathrm{dom}(f)
#     \end{equation*}
#     
# - die Menge aller Subgradienten von $f$ an der Stelle $x$ bezeichnen wir mit    
#     \begin{equation*} 
#     \partial f(x) = \big\{g\ |\ g\ \text{ist Subgradient von}\ f\ \text{in}\ x\big\}
#     \end{equation*}
#     
# 
# **Beispiel:** Ist $f(x)=|x|$, dann ist $\partial f(0) = [-1,1]$, denn
# \begin{equation*} 
# |y| = f(y) \geq f(0) + g(y-0) = gy \quad \forall y\in\mathbb{R}
# \end{equation*}
# genau dann wenn
# \begin{equation*} 
# g \in [-1,1].
# \end{equation*}
#   
#   
# **Lemma**: Ist $f$ differenzierbar in $x$, dann gilt $\partial f(x) \subset \{f'(x)\}$, also
#   $\partial f(x) = \{f'(x)\}$ oder $\partial f(x) = \emptyset$.
#   

# Die Ungleichung $f(y) \geq f(x) + g^T(y-x)$ sieht aus wie die Konvexitätsbedingung für $C^1$-Funktionen,
#   nur dass $f'(x)$ durch $g$ ersetzt wurde. 
# Dies ist kein Zufall.
# 
# 
# **Lemma**: $f:\mathbb{R}^d \supset \mathrm{dom}(f) \rightarrow \mathbb{R}$ ist konvex 
#   genau dann, wenn $\mathrm{dom}(f)$ konvex ist und
#   \begin{equation*} 
#   \partial f(x) \neq \emptyset
#   \quad
#   \forall x\in\mathrm{dom}(f).
#   \end{equation*}
#   

# **Beweis:**
# 
# "$\Rightarrow$"
# 
# - ist $f$ konvex, so ist $\mathrm{epi}(f)$ konvex 
#   
# - $\partial f(x) \neq \emptyset$ folgt aus dem
#  [Existenzsatz für Stützhyperebenen](https://de.wikipedia.org/wiki/St%C3%BCtzhyperebene) für konvexe Mengen angewandt auf $\mathrm{epi}(f)$
# 
# "$\Leftarrow$"
# 
# - nach Voraussetzung ist $\mathrm{dom}(f)$ konvex 
#   und $\partial f(x) \neq \emptyset$, d.h.
#   es existiert $g(x) \in \partial f(x)$ mit
# \begin{equation*} 
# f(y) \geq f(x) +g(x) (y-x) \quad \forall x,y\in \mathrm{dom}(f)
# \end{equation*}
# 
# - für $t\in \left[ 0,1\right]$ setzen wir
# \begin{equation*} 
# z= (1-t) x+ty
# \end{equation*}
# 
# - dann ist $z\in \mathrm{dom}(f)$ und
# \begin{equation*} 
# \begin{aligned}
# f(x) 
# &\geq f(z) + g(z) (x-z) 
# \\ 
# f(y) 
# &\geq f(z) + g(z) (y-z)
# \end{aligned}
# \end{equation*}
# 
# - multiplizieren wir die erste Ungleichung mit $1-t$, die zweite mit $t$ und addieren die Ergebnisse, dann erhalten wir
# \begin{align*}
#  (1-t) f(x) + t f(y) 
# &\geq (1-t)\big(f(z) +g(z)  (x-z) \big) 
# \\
# & \quad + t \big(f(z) +g(z) (y-z)\big)\\
# &= f(z) +g(z) \big( \underbrace{( 1-t) x+ty}_z -z\big)\\
# &= f(z) \\
# &= f\big(( 1-t) x+ty \big)
# \end{align*}
# 
# $\square$

# Existenz von Subgradienten und Konvexität sind also äquivalent.

# Einige weitere Eigenschaften des Gradienten lassen sich auch auf Subgradienten übertragen.
# Für Lipschitz-Stetigkeit bei $C^1$-Funktionen $f$ hatten wir $\|f'(x)\|\leq L_f$.
# Dies gilt analog auch für Subgradienten.
# 
# 
# **Lemma**: $f:\mathbb{R}^d \supset \mathrm{dom}(f) \rightarrow \mathbb{R}$ konvex, $\mathrm{dom}(f)$ offen.
#   Dann ist
#   \begin{equation*} 
#   \|f(x) - f(y)\| \leq L_f \|x - y\| \quad \forall x,y\in \mathrm{dom}(f)
#   \end{equation*}
# äquivalent zu
#   \begin{equation*} 
#   \|g\| \leq L_f \quad \forall g \in \partial f(x) \quad \forall x\in \mathrm{dom}(f).
#   \end{equation*}

# **Beweis**: 
# 
# "$\Rightarrow$"
# 
#   - für $x \in \mathrm{dom}(f)$, $g \in \partial f(x)$, $\varepsilon>0$ definieren wir
#   \begin{equation*} 
#     y = 
#     \begin{cases}
#     x + \varepsilon \frac{g}{\|g\|} & g \neq 0\\
#     x & g=0
#     \end{cases}
#     \end{equation*}
#   
# - damit folgt
#     \begin{equation*} 
#     y -x = 
#     \begin{cases}
#     \varepsilon \frac{g}{\|g\|} & g \neq 0\\
#     0 & g=0
#     \end{cases}
#     \end{equation*}
#   bzw.
#     \begin{equation*} 
#     \|y-x\|
#     = 
#     \begin{cases}
#     \varepsilon & g \neq 0\\
#     0 & g=0 
#     \end{cases}
#     \leq \varepsilon
#     \end{equation*}
#   
# - wegen $\mathrm{dom}(f)$ offen ist für $\varepsilon>0$ hinreichend klein 
#   auch $y\in\mathrm{dom}(f)$
#   
# - außerdem gilt
#     \begin{equation*} 
#     g^T(y-x)
#     = 
#     \begin{cases}
#     \varepsilon\frac{g^T g}{\|g\|} & g \neq 0\\
#     0 & g=0 
#     \end{cases}
#     = \varepsilon\|g\|
#     \end{equation*}
#   
# - wegen $g \in \partial f(x)$ ist dann
#     \begin{equation*} 
#     f(y) \geq f(x) + g^T(y-x) = f(x) + \varepsilon\|g\| 
#     \end{equation*}
#   
# - da $f$ Lipschitz-stetig ist folgt
#     \begin{equation*} 
#     L_f \varepsilon \geq L_f \|y-x\| \geq f(y) - f(x) \geq \varepsilon\|g\|, 
#     \end{equation*}
#   also
#     \begin{equation*} 
#     \|g\| \leq L_f
#     \end{equation*}
#     
# "$\Leftarrow$"
#   
# - für $x,y \in \mathrm{dom}(f)$ und $g \in \partial f(x)$ gilt
#     \begin{equation*} 
#     f(y) \geq f(x) + g^T(y-x)
#     \end{equation*}
#   also
#     \begin{equation*} 
#     f(x) - f(y) \leq  g^T(x-y)
#     \end{equation*}
#     
# - mit Cauchy-Schwartz und $\|g\| \leq L_f$ folgt
#     \begin{equation*} 
#     f(x) - f(y) \leq  \|g\| \|x-y\| \leq L_f \|x -y\| 
#     \end{equation*}    
#     
# - Vertauschen von $x$ und $y$ liefert schließlich
#     \begin{equation*} 
#     |f(x) - f(y)| \leq L_f \|x -y\| \quad \forall x,y \in \mathrm{dom}(f)
#     \end{equation*}    
#     
# $\square$ 

# **Satz**: $f:\mathbb{R}^d \supset \mathrm{dom}(f) \rightarrow \mathbb{R}$, $x \in \mathrm{dom}(f)$, dann gilt
#   \begin{equation*} 
#   0 \in \partial f(x) 
#   \quad\Leftrightarrow\quad
#   x\ \text{ist globales Minimum von}\ f.
#   \end{equation*}

# **Beweis**: 
#   
# "$\Rightarrow$"
#   
# - ist $0 \in \partial f(x)$, dann folgt aus der Definition des Subgradienten
#     \begin{equation*} 
#     f(y) \geq f(x) + 0^T(y-x) =  f(x) \quad \forall y\in\mathrm{dom}(f)
#     \end{equation*}
#     
# "$\Leftarrow$"
#   
# - ist $x \in\mathrm{dom}(f)$ globales Minimum, dann gilt
#     \begin{equation*} 
#     f(y) \geq f(x) = f(x) + 0^T(y-x)  \quad \forall y\in\mathrm{dom}(f)
#     \end{equation*}
# 
# $\square$ 

# **Bemerkung:**
# 
# - die Bedingung $0 \in \partial f(x)$ ist stärker als $f'(x)=0$ bei $C^1$-Funktionen
#   
# - $f'(x)=0$ ist nur eine *notwendige Bedingung* für *lokale* Minima, d.h.
#     aus $f'(x)=0$ alleine kann man i.A. nicht folgern, dass $x$ ein lokales
#     oder gar globales Minimum ist
#     
# - dagegen garantiert $0 \in \partial f(x)$ immer, dass $x$ globales Minimum ist

# ## Vorüberlegungen

# Wir betrachten nun *Subgradient-Descent*
#   \begin{equation*} 
#   x_{t+1} = x_t - \gamma_t g_t, \quad g_t \in \partial f(x_t).
#   \end{equation*}
# Dabei ist $g_t$ *irgendein* Element aus $\partial f(x_t)$.
# 
# Ist $f$ konvex, dann ist $\partial f(x) \neq \emptyset$ $\forall x$,
#   so dass immer mindestens ein $g_t$ existiert.
# 
# Ist $f$ konvex und in $x_t$ differenzierbar, dann gilt
#   \begin{equation*} 
#   \partial f(x_t) = \{ f'(x_t)\}
#   \end{equation*}
# und somit
#   \begin{equation*} 
#   g_t = f'(x_t).
#   \end{equation*}
# Wir lassen auch variable Schrittweite $\gamma_t$ zu.

# Unter diesen Voraussetzungen wollen wir analog zu Gradient-Descent die Konvergenzeigenschaften untersuchen.
# Viele Ergebnisse lassen sich direkt übertragen, indem man einfach $f'_t$ durch $g_t$ ersetzt.
# 
# Für Schrittweite $\gamma$ erhalten wir damit
#   \begin{equation*} 
#   g_{t}^T\left(x_{t}-x_{*}\right) 
#   =\frac{1}{2 \gamma}\big(\gamma^{2}\|g_{t}\|_{2}^{2}+\|x_{t}-x_{*}\|_{2}^{2}-\|x_{t+1}-x_{*}\|_{2}^{2}\big).
#   \end{equation*}
#   
# Da $g_t$ Subgradient in $x_t$ ist, gilt
#   \begin{equation*} 
#   f_\ast \geq f_t + g_t^T (x_\ast - x_t)
#   \end{equation*}
# und somit
#   \begin{align*} 
#   f_t - f_\ast 
#   &\leq g_t^T (x_t - x_\ast)
#   \\
#   &=\frac{1}{2 \gamma}\big(\gamma^{2}\|g_{t}\|_{2}^{2}+\|x_{t}-x_{*}\|_{2}^{2}-\|x_{t+1}-x_{*}\|_{2}^{2}\big).
#   \end{align*}

# ## Lipschitz-Stetigkeit

# Lipschitz-Stetigkeit von $f$, d.h.
#   \begin{equation*} 
#   \|f(y) - f(x)\| \leq L_f \|y - x\| \quad \forall x,y\in\mathbb{R}^d
#   \end{equation*}
# ist äquivalent zu
#   \begin{equation*} 
#   \|g\| \leq L_f  \quad \forall g\in \partial f(x) \quad \forall x\in\mathbb{R}^d.
#   \end{equation*}
# Damit erhalten wir aus der letzten Ungleichung im vorherigen Abschnitt
#   \begin{equation*} 
#   f_t - f_\ast 
#   \leq \frac{1}{2 \gamma}\big(\gamma^{2}L_f^{2}+\|x_{t}-x_{*}\|_{2}^{2}-\|x_{t+1}-x_{*}\|_{2}^{2}\big)
#   \end{equation*}
# und somit
#   \begin{align*} 
#   \sum_{t=0}^{T-1} (f_t - f_\ast)
#   & \leq 
#   \frac{\gamma}{2}T L_f^2 
#   + \frac{1}{2\gamma} 
#   \big( 
#   \underbrace{\|x_{0}-x_{*}\|_{2}^{2}}_{e_0^2}
#   -
#   \underbrace{\|x_{T}-x_{*}\|_{2}^{2}}_{\geq 0}
#   \big)
#   \\
#   & \leq \frac{\gamma T L_f^2}{2} + \frac{e_0^2}{2\gamma},
#   \end{align*}
# was identisch ist mit der Abschätzung bei Gradient-Descent.
#   
# Somit erhalten wir in diesem Fall auch die selbe Konvergenzaussage
#   wie bei Gradient-Descent.
# 

# **Satz:** $f:\mathbb{R}^d\to \mathbb{R}$, konvex, L-stetig mit Konstante $L_f$
#   und es existiere $x_\ast = \mathrm{argmin}_{x\in\mathbb{R}^d}f(x)$.
# Mit $\gamma = \frac{c}{T^\omega}$, $\omega\in(0,1)$, gilt
#   \begin{align*} 
#   \min_{t=0,\ldots,T-1}(f_t - f_\ast)
#   &\leq \frac{1}{T} \sum_{t=0}^{T-1} (f_t - f_\ast)\\
#   &= \mathcal{O}\Big(\big(\frac{1}{T}\big)^{\min(\omega,1-\omega)}\Big)
#   \quad 
#   \text{für}
#   \quad
#   T\to\infty.
#   \end{align*}
# Die optimale Ordnung ist $\frac{1}{2}$ für $\omega=\frac{1}{2}$.
# Mit $e_0 = \|x_0 - x_\ast\|$, $\gamma = \frac{e_0}{L_f\sqrt{T}}$ gilt außerdem
#   \begin{equation*} 
#   \min_{t=0,\ldots,T-1}(f_t - f_\ast)
#   \leq \frac{1}{T} \sum_{t=0}^{T-1} (f_t - f_\ast)
#   \leq \frac{L_f e_0}{\sqrt{T}}.
#   \end{equation*}  

# ## $L$-Glattheit

# Bei differenzierbarem $f$ hatten wir unter zusätzlichen   Voraussetzungen ($L$-Glattheit, $\mu$-Konvexität) höhere Konvergenzraten nachweisen können.
# 
# Bei $L$-Glattheit stoßen wir hier auf ein Problem.
# Wie das folgende Lemma zeigt, sind $L$-glatte Funktionen mit existierenden Subgradienten automatisch differenzierbar.
#   
#   
# **Lemma:**
# $f:\mathbb{R}^d \supset \mathrm{dom}(f) \to \mathbb{R}$,
#   $\mathrm{dom}(f)$ offen und in $x\in \mathrm{dom}(f)$
#   gelte $\partial f(x)\neq\emptyset$.
# Gibt es ein $L\ge 0$ so dass für $g_x \in \partial f(x)$
#   \begin{equation*} 
#   f(y) \leq f(x) + g_x^T (y-x) + \frac{L}{2}\|y-x\|_2^2
#   \quad \forall y\in\mathrm{dom}(f)
#   \end{equation*}
#   gilt, dann ist $f$ differenzierbar in $x$ und $f'(x)=g_x$.
#   
#   
# **Beweis:**
# 
# - wegen $g_x \in \partial f(x)$ gilt
#     \begin{equation*} 
#     f(y) \geq f(x) + g_x^T(y-x) \quad \forall y\in\mathrm{dom}(f)
#     \end{equation*}
#   und es folgt
#     \begin{equation*} 
#     0 \leq f(y) - \big(f(x) + g_x^T (y-x)\big) \leq \frac{L}{2} \|y-x\|_2^2
#     \end{equation*}
#     
# - also ist für $y \to x$
#     \begin{equation*} 
#     \big| f(y) - \big(f(x) + g_x^T (y-x)\big) \big|= {\scriptstyle \mathcal{O}}(\|y-x\|)
#     \end{equation*}
#   und somit $f$ in $x$ differenzierbar mit Ableitung $f'(x)=g_x$
#     
# $\square$  
#   
#   
# Verlangen wir also $L$-Glattheit für subdifferenzierbare
# Funktionen (z.B. $f$ konvex), so landen wir wieder beim differenzierbarem Fall.

# ## $\mu$-Konvexität

# $\mu$-Konvexität lässt sich einfach auf subdifferenzierbare Funktionen verallgemeinern.
# 
# 
# **Definition:**
# $f:\mathbb{R}^d \supset \mathrm{dom}(f) \to \mathbb{R}$
#   heißt $\mu$-konvex, falls
#   \begin{equation*} 
#   f(y) \geq f(x) + g^T (y-x) + \frac{\mu}{2}\|y-x\|_2^2
#   \quad \forall g \in \partial f(x)
#   \quad \forall x,y\in\mathrm{dom}(f).
#   \end{equation*}
#   
# 
# Wie im letzten Kapitel erklärt, müssen wir bei nicht differenzierbarem $f$ auf $L$-Glattheit verzichten, was einige Schwierigkeiten verursachen wird.
# 
# Die Probleme die sich dabei ergeben werden wir am folgenden Beispiel näher untersuchen.
# 
# 
# **Beispiel:**
# 
# - für die Funktion $f:\mathbb{R}\to\mathbb{R}$
#   \begin{equation*} 
#   f(x) 
#   = e^{|x|}
#   = \begin{cases}
#     e^{-x} & x<0\\
#     e^{x} & 0 \leq x
#     \end{cases}
#   \end{equation*}
#   erhalten wir als Subgradient
#   \begin{equation*} 
#   \partial f(x) 
#   = \begin{cases}
#     \mathrm{sign}(x)\ e^{|x|} & x \neq 0\\
#     [-1,1] & x = 0
#     \end{cases}
#   \end{equation*}
#   
# - $f$ ist $\mu$-konvex mit $\mu=1$ (Übung)
#   
# - $\mu$-Konvexität liefert nur eine Schranke nach unten, d.h. $f$
#     kann sehr stark wachsen und somit kann $\partial f$ sehr große 
#     Werte annehmen
#   
# - ist dies der Fall, dann können bei Subgradient-Descent 
#     "Overshoots" auftreten
#     
# - um dies zu kompensieren und Konvergenz zu sichern, muss
#     die Schrittweite $\gamma$ sehr klein gewählt werden
#     
# - für "brave" Funktionen $f$ verursacht das aber unnötigen Aufwand
#   
# - als Konsequenz muss die Schrittweite entsprechend an $f$ angepasst 
#     werden

# Beim einfachen Gradient-Descent hat die $L$-Glattheit (als Beschränkung nach oben) dieses Problem beseitigt.
# Da  $L$-Glattheit hier nicht zur Verfügung steht, werden wir
#   unter der zusätzlichen Annahme 
#   \begin{equation*} 
#   \|g_t\| \leq B \quad \forall t
#   \end{equation*}
# und $t$-abhängiger Schrittweite $\gamma_t$
#   das folgende Konvergenz-Resultat für Subgradient-Descent
#   beweisen.
#   
#   
# **Satz:** $f$ sei $\mu$-konvex und es existiere $x_\ast$.
# Für
#   \begin{equation*} 
#   \gamma_t = \frac{2}{\mu(t+1)}
#   \end{equation*}
# erhalten wir für Subgradient-Descent
#   \begin{equation*} 
#   f\Big(
#   \underbrace{\frac{2}{T(T+1)} \sum_{t=1}^T t x_t}
#    _{\text{Konvexkombination der }x_t}
#   \Big) - f_\ast
#   \leq
#   \frac{2 B^2}{\mu(T+1)}
#   \end{equation*}
# mit
#   \begin{equation*} 
#   B = \max_{t=1,\ldots,T}\|g_t\|_2.
#   \end{equation*}

# **Beweis:**
# 
# - aus den Vorüberlegungen wissen wir
#     \begin{equation*} 
#     g_{t}^T(x_{t}-x_{*}) 
#     =\frac{1}{2 \gamma_t}
#     \big(
#     \gamma_t^{2}\|g_{t}\|_{2}^{2}
#     +\|x_{t}- x_{*}\|_{2}^{2}
#     -\|x_{t+1}-x_{*}\|_{2}^{2}
#     \big)
#     \quad 
#     \end{equation*}
#    
# - aus der $\mu$-Konvexität von $f$ folgt mit $y=x_\ast$, $x=x_t$
#     \begin{equation*} 
#     f_\ast 
#     \geq 
#     f_t + g_t^T (x_\ast-x_t) + \frac{\mu}{2}\|x_\ast-x_t\|_2^2
#     \end{equation*}
#   also
#     \begin{equation*} 
#     f_t - f_\ast  
#     \leq 
#     g_t^T (x_t-x_\ast)
#     - \frac{\mu}{2}\|x_t-x_\ast\|_2^2
#     \end{equation*}
#   und somit
#     \begin{align*}
#     f_t - f_\ast
#     &\leq
#      \frac{1}{2 \gamma_t}
#      \big(
#      \gamma_t^{2}\|g_{t}\|_{2}^{2}
#      +\|x_{t}- x_{*}\|_{2}^{2}
#      -\|x_{t+1}-x_{*}\|_{2}^{2}
#      \big)
#      - \frac{\mu}{2}\|x_t-x_\ast\|_2^2 \\
#     &\leq
#      \frac{B^2}{2}\gamma_t
#      + \frac{1}{2}
#        \big(\frac{1}{\gamma_t} - \mu\big) \|x_{t}- x_{*}\|_{2}^{2}
#      - \frac{1}{2 \gamma_t} \|x_{t+1}-x_{*}\|_{2}^{2}
#     \end{align*}
#    
# - multiplizieren wir mit $t$ und setzen wir wieder
#     $e_t = \|x_{t}- x_{*}\|_{2}$, so erhalten wir
#     \begin{equation*} 
#     t(f_t - f_\ast)
#     \leq
#     \frac{t}{2} 
#     \Big(
#     B^2 \gamma_t
#     + \underbrace{\big(\frac{1}{\gamma_t} - \mu\big)}_{\alpha_t} \ e_t^2
#     - \underbrace{\frac{1}{\gamma_t}}_{\beta_t} \ e_{t+1}^2
#     \Big)
#     \end{equation*}
#     
# - Summation über $t$ liefert
#     \begin{align*}
#     \sum_{t=1}^T t(f_t - f_\ast)
#     & = 
#     \sum_{t=0}^T t(f_t - f_\ast)\\
#     & \leq
#     \sum_{t=0}^T
#     \frac{t}{2} 
#     \Big(B^2 \gamma_t + \alpha_t e_t^2 - \beta_t e_{t+1}^2\Big)\\
#     & = 
#     \frac{B^2}{2} \sum_{t=1}^T t \gamma_t
#     + \frac{1}{2}
#     \Big(
#     \sum_{t=0}^T t \alpha_t e_t^2  - \sum_{t=0}^T t \beta_t e_{t+1}^2
#     \Big)\\
#     & = 
#     \frac{B^2}{2} \sum_{t=1}^T t \gamma_t
#     + \frac{1}{2}
#     \Big(
#       \sum_{t=0}^{T-1} (t+1) \alpha_{t+1} e_{t+1}^2  
#     - \sum_{t=0}^T t \beta_t e_{t+1}^2
#     \Big)\\
#     & = 
#     \frac{B^2}{2} \sum_{t=1}^T t \gamma_t
#     + \frac{1}{2}
#     \Big(
#       \sum_{t=0}^{T-1} \big((t+1) \alpha_{t+1} - t \beta_t\big) e_{t+1}^2  
#     - \beta_T e_{T+1}^2   
#     \Big)\\
#     & \leq 
#     \frac{B^2}{2} \sum_{t=1}^T t \gamma_t
#     + \frac{1}{2}
#     \Big(
#       \sum_{t=0}^{T-1} \big((t+1) \alpha_{t+1} - t \beta_t\big) e_{t+1}^2  
#     \Big)
#     \end{align*}    
#     
# - $\alpha_t$ und $\beta_t$ hängen nur von $\gamma_t$ und $\mu$ ab
# 
# - kann man $\gamma_t$ so wählen, dass
#   \begin{equation*} 
#   (t+1) \alpha_{t+1} - t \beta_t \leq 0
#   \quad \forall t \geq 0
#   \end{equation*}
#   gilt, dann kann der zweite Summand auf der rechten Seite mit $0$
#   nach oben abgeschätzt werden
#   
# - für
#   \begin{equation*} 
#   \gamma_t = \frac{2}{\mu(t+1)}
#   \end{equation*}
#   ist
#   \begin{align*}
#   (t+1) \alpha_{t+1} - t \beta_t 
#   &= (t+1) \big(\frac{1}{\gamma_{t+1}} - \mu\big) - t \frac{1}
#      {\gamma_t}\\
#   &= (t+1) \big(\frac{\mu(t+2)}{2} - \mu\big) - t \frac{\mu(t+1)}{2}\\
#   &= \frac{\mu}{2} (t+1)t - t\frac{\mu}{2} (t+1)\\
#   &= 0
#   \end{align*}
#   und damit
#   \begin{equation*} 
#   \sum_{t=1}^T t(f_t - f_\ast) 
#   \leq 
#   \frac{B^2}{2} \sum_{t=1}^T t \gamma_t
#   \end{equation*}
#   
# - wegen
#   \begin{equation*} 
#   \frac{2}{T(T+1)}\sum_{t=1}^T t = 1,
#   \end{equation*}
#   der Konvexität von $f$ und der Jensen-Ungleichung
#   erhalten wir 
#   \begin{align*}
#   f\Big(\frac{2}{T(T+1)}\sum_{t=1}^T t x_t \Big) - f_\ast
#   &\leq \Big(\frac{2}{T(T+1)}\sum_{t=1}^T t f_t \Big) - f_\ast\\
#   &= \frac{2}{T(T+1)}\sum_{t=1}^T t (f_t - f_\ast)\\
#   &\leq \frac{B^2}{T(T+1)} \sum_{t=1}^T t \gamma_t
#   \end{align*} 
#   
# - mit
#   \begin{equation*} 
#   \gamma_t = \frac{2}{\mu(t+1)}
#   \end{equation*}
#   folgt
#   \begin{equation*} 
#   \sum_{t=1}^T t \gamma_t
#   = \frac{2}{\mu} \sum_{t=1}^T \frac{t}{t+1} \leq \frac{2}{\mu} T
#   \end{equation*}
#   und somit schließlich
#   \begin{equation*} 
#   f\Big(\frac{2}{T(T+1)}\sum_{t=1}^T t x_t \Big) - f_\ast
#   \leq \frac{2B^2}{\mu(T+1)}
#   \end{equation*}
#   
# $\square$

# **Bemerkung:**
# 
# - in der oberen Schranke steckt $x_0$ nicht explizit drin, geht aber
#     implizit über $B$ ein ($\|g_t\|_2 \leq B$ $\forall t$)
#    
# - für $f$ differenzierbar, $\mu$-konvex und $L$-glatt hatten wir bei
#     Gradient-Descent die Komlplexität $\mathcal{O}\big(\log(\frac{1}{\varepsilon})\big)$
#    
# - für $f$ $\mu$-konvex, $\|g_t\|_2 \leq B$ $\forall t$ bekommen wir
#     bei Subgradient-Descent nur die Komlplexität $\mathcal{O}\big(\frac{1}{\varepsilon}\big)$, 
#     d.h. aus fehlender Glattheit von $f$ kann wieder langsemere Konvergenz folgen

# ## Zusammenfassung

# Ist $f$ konvex und nicht differenzierbar, dann benutzt man bei Gradient-Descent statt des 
#   (eventuell nicht  existierenden) Gradienten einen Subgradienten.
#   
# Für das *nicht restringierte Optimierungsproblem* haben wir folgendes 
#   Konvergenzverhalten nachgewiesen:
# 
#   - $f$ konvex und Lipschitz-stetig, $\gamma = \frac{c}{\sqrt{T}}$, $c>0$:
#     \begin{equation*} 
#     \min_{t=0,\ldots,T-1}(f_t - f_\ast)
#     \leq \frac{L_f e_0}{\sqrt{T}}
#     \end{equation*}  
#     also
#     \begin{equation*} 
#     \min_{t=0,\ldots,T-1}(f_t - f_\ast) \leq \varepsilon
#     \quad \Rightarrow\quad
#     T = \mathcal{O}\big(\frac{1}{\varepsilon^2}\big)
#     \end{equation*}
#     
#   - $f$ $\mu$-konvex mit $\mu>0$ und $\|g_t\|\leq B$,
#     $\gamma_t = \frac{2}{\mu(t+1)}$
#     \begin{equation*} 
#     f\Big(\frac{2}{T(T+1)} \sum_{t=1}^T t x_t\Big) - f_\ast
#     \leq
#     \frac{2 B^2}{\mu(T+1)},
#     \end{equation*}
#     also
#     \begin{equation*} 
#     f\Big(\frac{2}{T(T+1)} \sum_{t=1}^T t x_t\Big) - f_\ast
#     \leq \varepsilon
#     \quad \Rightarrow\quad
#     T = \mathcal{O}\Big( \frac{1}{\varepsilon} \Big)
#     \end{equation*}
#     
# Im ersten Fall benötigen wir für Genauigkeit $\varepsilon$
#   den selben asymptotischen Aufwand wie bei 
#   differenzierbarem $f$.
#   
# Im zweiten Fall steigt der
#   Aufwand von $\mathcal{O}\Big( \log\big(\frac{1}{\varepsilon}\big) \Big)$
#   auf $\mathcal{O}\Big( \frac{1}{\varepsilon} \Big)$, d.h. durch die
#   reduzierten Glattheitsanforderungen an $f$ reduziert sich hier
#   die Konvergenzgeschwindigkeit.
