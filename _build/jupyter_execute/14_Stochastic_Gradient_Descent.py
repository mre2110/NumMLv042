#!/usr/bin/env python
# coding: utf-8

# # Stochastic Gradient Descent

# ## Überblick

# In viele Anwendungen hat $f$ die Struktur
#   \begin{equation*} 
#   f(x) = \frac{1}{n} \sum_{i=1}^n f_i(x),
#   \quad
#   f_i \text{ differenzierbar},
#   \end{equation*}
# z.B. wenn $f_i$ die Loss-Funktion zum $i$-ten Trainings-Datensatz ist.
#   
# Bei einfachem Gradient-Descent muss in jedem Schritt
#   \begin{equation*} 
#   f'(x) = \frac{1}{n} \sum_{i=1}^n f'_i(x),
#   \end{equation*}
#   berechnet werden.
# Dies wird nun wie folgt vereinfacht:
# 
#  - wähle in jedem Schritt zufällig (gleichverteilt) ein $i_t\in\{1,\ldots,n\}$ aus
#   
#  - setze
#    \begin{equation*} 
#    x_{t+1} = x_t - \gamma_t f'_{i_t}(x_t)
#    \end{equation*}
#    
# Man bezeichnet
#    \begin{equation*} 
#    g_t = g(i_t, x_t) = f'_{i_t}(x_t)
#    \end{equation*}
# als *stochastischen Gradienten*.
# Der Aufwand pro Schritt wird im Vergleich zum einfachen Gradient-Descent-Verfahren um
#    den Faktor $n$ reduziert.

# ## Vorüberlegungen

# Eine direkte Übertragung der Konvergenzanalysen des Gradient-Descent-Verfahrens funktioniert zunächst nicht.
# 
# Bei Gradient-Descent hatten wir aus der Konvexitätsbedingung
#   \begin{equation*} 
#   f(y) \geq f(x) + f'(x)(y-x)
#   \end{equation*}
# die Ungleichung
#   \begin{equation*} 
#   f_t - f_\ast \leq f_t' (x_t - x_\ast)
#   \end{equation*}
# abgeleitet und die weiteren Untersuchungen darauf aufgebaut.
# 
# Bei Stochastic-Gradient-Descent geht das nicht, da
#   $g_t \neq f'(x_t)$.
# Wir können aber zeigen, dass die Ungleichung für Erwartungswerte
#   gilt:
#   
# - ist $i_t\in\{1,\ldots,n\}$ eine gleichverteilte Zufallsvariable, dann gilt
#   \begin{equation*} 
#   \mathbb{P}(i_t=k) = \frac{1}{n} \quad \forall k\in\{1,\ldots,n\}
#   \end{equation*}
#   und somit
#   \begin{align*}
#   \mathbb{E}_{i_t}(g_t)
#   &= \sum_{k=1}^n f'_{i_t}(x_t)\mathbb{P}(i_t=k)\\
#   &= \frac{1}{n}\sum_{k=1}^n f'_{i_t}(x_t)\\
#   &= f'(x_t)
#   \end{align*}
#   
# - $g_t$ ist also ein erwartungstreuer Schätzer von $f'(x_t)$

# Wir beginnen mit einer grundlegenden Abschätzung für Stochastic-Gradient-Descent
#   \begin{equation*} 
#   x_{t+1} = x_t - \gamma \underbrace{g(i_t, x_t)}_{g_t}.
#   \end{equation*}
#   
#   
# **Lemma:** Ist $f'$ Lipschitz-stetig mit Konstante $L$, dann gilt
#   \begin{equation*} 
#   \mathbb{E}_{i_t}(f_{t+1}) - f_t
#     \leq
#   - \gamma f'_t \ \mathbb{E}_{i_t}(g_t) 
#   + \frac{1}{2} \gamma^2 L \ \mathbb{E}_{i_t}(\|g_t\|_2^2).
#   \end{equation*}

# **Beweis:**
# 
# - ist $f'$ Lipschitz-stetig dann gilt (siehe Gradient-Descent) 
#     \begin{equation*} 
#     f(y) \leq f(x) + f'(x)(y-x) + \frac{1}{2} L \|y-x\|_2^2
#     \end{equation*}
#    
# - mit $y=x_{t+1}$, $x=x_t$ erhalten wir
#     \begin{equation*} 
#     f_{t+1} - f_t \leq f'_t(x_{t+1}-x_t) + \frac{1}{2} L \|x_{t+1}-x_t\|_2^2
#     \end{equation*}
#    
# - wegen $x_{t+1} - x_t = - \gamma g_t$ ist
#     \begin{equation*} 
#     f_{t+1} - f_t \leq -\gamma f'_t g_t + \frac{1}{2} \gamma^2 L   \|g_t\|_2^2
#     \end{equation*}
#    
# - bilden wir auf beiden Seiten $\mathbb{E}_{i_t}$ und beachten wir, dass
#     $f_t$, $f'_t$ nicht von $i_t$ abhängen, dann folgt
#     \begin{equation*} 
#     \mathbb{E}_{i_t}(f_{t+1}) - f_t
#     \leq
#     - \gamma f'_t \ \mathbb{E}_{i_t}(g_t) 
#     + \frac{1}{2} \gamma^2 L \ \mathbb{E}_{i_t}(\|g_t\|_2^2)
#     \end{equation*}
#    
# $\square$

# **Bemerkung:** 
#  
# - der erwartete Abstieg ist beschränkt durch
#     den Erwartunsgwert der Richtungsableitung von $f$ in Richtung $g_t$
#     \begin{equation*} 
#     f'_t \ \mathbb{E}_{i_t}(g_t)
#     =
#     \mathbb{E}_{i_t}(g_t)^T f'_t 
#     \end{equation*}
#   sowie
#     \begin{equation*} 
#     \mathbb{E}_{i_t}(\|g_t\|_2^2)
#     \end{equation*}
#    
# - wie man zu $x_t$ gekommen ist, spielt keine Rolle (Markov-Eigenschaft)
#   
# - ist $g_t$ ein erwartungstreuer Schätzer, d.h. $\mathbb{E}_{i_t}(g_t)=f'_t$,
#     dann folgt
#     \begin{align*}
#     \mathbb{E}_{i_t}(f_{t+1}) - f_t
#     & \leq
#     - \gamma f'_t \ \mathbb{E}_{i_t}(g_t) 
#     + \frac{1}{2} \gamma^2 L \ \mathbb{E}_{i_t}(\|g_t\|_2^2) \\
#     &\leq 
#     - \gamma \|f'_t\|_2^2
#     + \frac{1}{2} \gamma^2 L \ \mathbb{E}_{i_t}(\|g_t\|_2^2)
#     \end{align*}
#     
# - kann $\mathbb{E}_{i_t}(\|g_t\|_2^2)$ "deterministisch" beschränkt werden, dann
#     erhalten wir wieder einen hinreichenden Abstieg
#     
#     
# **Definition:** Es sei
#   \begin{equation*} 
#   \mathbb{V}_{i_t} = \mathbb{E}_{i_t}(\|g_t\|_2^2) - \|\mathbb{E}_{i_t}(g_t) \|_2^2.
#   \end{equation*}

# Für den Rest des Kapitels treffen wir folgende Annahmen:
# 
# - $x_t \in X$, $X$ offen, $f|_X \geq \bar{f}$
#  
# - $\exists c_G \geq c > 0$ so dass $\forall t$ gilt
#     \begin{equation*} 
#     f_t \ \mathbb{E}_{i_t}(g_t) \geq c \| f'_t \|_2^2,
#     \quad
#     \| \mathbb{E}_{i_t}(g_t) \|_2 \leq c_G \| f'_t \|_2
#     \end{equation*}
#    
# - $\exists M,M_V \geq 0$ mit
#     \begin{equation*} 
#     \mathbb{V}_{i_t} \leq M + M_V \| f'_t \|_2^2
#     \end{equation*}
#    
# 
# **Bemerkung:** Ist $g_t$ ein erwartungstreuer Schätzer von $f'_t$, dann
#   ist $c=c_G=1$.
#   
# 
# Aus der Annahme folgt nun
#   \begin{align*}
#     \mathbb{E}_{i_t}(\|g_t\|_2^2)
#     &  =   \mathbb{V}_{i_t} +  \| \mathbb{E}_{i_t}(g_t) \|_2^2\\
#     &\leq  M + M_V \| f'_t \|_2^2 + c_G^2 \| f'_t \|_2^2 \\
#     &\leq  M + \underbrace{(M_V + c_G^2)}_{M_G \geq 0} \| f'_t \|_2^2 .
#   \end{align*}
# Kombiniert mit dem letzten Lemma erhalten wir
#   \begin{align*}
#   \mathbb{E}_{i_t}(f_{t+1}) - f_t
#   &\leq
#     - \gamma f'_t \ \mathbb{E}_{i_t}(g_t) 
#     + \frac{1}{2} \gamma^2 L \ \mathbb{E}_{i_t}(\|g_t\|_2^2)\\
#   &\leq -\gamma c \| f'_t \|_2^2 
#         +\frac{1}{2} \gamma^2 L \big(M + M_G \| f'_t \|_2^2 \big) \\
#   &= -\gamma \big(c - \frac{\gamma L}{2} M_G \big) \| f'_t \|_2^2 
#      + \frac{1}{2} \gamma^2 L M
#   \end{align*}
# und somit
# 
# 
# **Lemma:**
#   \begin{equation*} 
#   \mathbb{E}_{i_t}(f_{t+1}) - f_t 
#   \leq
#      -\gamma \big(c - \frac{\gamma L}{2} M_G \big) \|  f'_t \|_2^2 
#      + \frac{1}{2} \gamma^2 L M
#   \end{equation*}
#   
# 
# **Bemerkung:**
# 
#   - die rechte Seite der Abschätzung ist deterministisch
#   
#   - die Schranke für $\mathbb{E}_{i_t}(f_{t+1}) - f_t$ hängt nur von $x_t$ und $i_t$
#     ab und *nicht* von früheren $x_s$, $s<t$
#     
#   - ist $\gamma$ klein genug, dann ist $\mathbb{E}_{i_t}(f_{t+1}) - f_t < 0$

# ## Konvergenz

# $f$ sei jetzt differenzierbar, $L$-glatt und $\mu$-konvex 
#   und zusätzlich soll $x_\ast$ mit $f_\ast = f(x_\ast) = \inf_x f(x)$ existieren.
# Wir werden nun zeigen, dass wir unter diesen Voraussetzungen bei konstanter
#   Schrittweite $\gamma$ zwar keine Konvergenz erhalten, aber zumindest in
#   eine Umgebung von $x_\ast$ gelangen, deren Größe asymptotisch proportional
#   zu $\gamma$ ist.
#   
# Wir verwenden im Folgenden die Notation
#   \begin{equation*} 
#   \mathbb{E}(f_{t}) = \mathbb{E}_{i_{0}}\ldots\mathbb{E}_{i_{t-1}}(f_t).
#   \end{equation*}
#   
#   
# **Satz:**
# Es sei $f$ differenzierbar, $L$-glatt, $\mu$-konvex 
#   und es existiere 
#   $x_\ast$ mit $f_\ast = f(x_\ast)= \inf_x f(x)$.
#   Ist
#   \begin{equation*} 
#   0 < \gamma \leq \frac{c}{L M_G}
#   \end{equation*}
# dann gilt
#   \begin{align*} 
#   \mathbb{E}(f_{t} - f_\ast) 
#   &\leq 
#   \frac{\gamma L M}{2 \mu c} 
#   + (1-\gamma \mu c)^{t-1} 
#     \big(\mathbb{E}(f_1 - f_\ast) - \frac{\gamma L M}{2 \mu c}\big)\\
#   &\xrightarrow{t \to \infty} \frac{\gamma L M}{2 \mu c}.
#   \end{align*}

# **Beweis:**
# 
# - aus dem letzten Lemma im vorherigen Abschnitt wissen wir
#    \begin{align*} 
#    \mathbb{E}_{i_t}(f_{t+1} - f_t)
#    &=
#    \mathbb{E}_{i_t}(f_{t+1}) - f_t 
#    \\
#    &\leq
#       -\gamma \big(c - \frac{\gamma L}{2} M_G \big) \|  f'_t \|_2^2 
#       + \frac{1}{2} \gamma^2 L M
#    \end{align*}
#   
# - wegen 
#   \begin{equation*} 
#   0 < \gamma \leq \frac{c}{L M_G}
#   \end{equation*}
#   ist
#   \begin{equation*} 
#   \frac{\gamma L}{2}M_G  \leq \frac{c}{2},
#   \end{equation*}
#   und deshalb
#   \begin{equation*} 
#   \mathbb{E}_{i_t}(f_{t+1}) - f_t 
#   \leq
#      - \frac{\gamma c}{2} \|  f'_t \|_2^2 
#      + \frac{1}{2} \gamma^2 L M
#   \end{equation*}
#   
# - oben hatten wir gesehen, dass für $\mu$-konvexe Funktionen auf $\mathbb{R}^d$
#   \begin{equation*} 
#   f(x) - f_\ast \leq \frac{1}{2\mu} \|f'(x)\|_2^2 
#   \end{equation*}
#   gilt, also
#   \begin{equation*} 
#   \|  f'_t \|_2^2  \geq 2\mu(f_t - f_\ast)
#   \end{equation*}
#   und somit
#   \begin{equation*} 
#   \mathbb{E}_{i_t}(f_{t+1}- f_\ast) - (f_t - f_\ast)
#   \leq
#      -\gamma\mu c (f_t - f_\ast)
#      + \frac{1}{2} \gamma^2 L M
#   \end{equation*}
#  
# - nun bilden wir auf beiden Seiten die Erwartungswerte 
#     $\mathbb{E}_{i_1} \ldots \mathbb{E}_{i_{t-1}}$ und
#   erhalten
#     \begin{equation*} 
#     \mathbb{E}(f_{t+1}- f_\ast) - \mathbb{E}(f_t - f_\ast)
#     \leq
#       -\gamma\mu c \mathbb{E}(f_t - f_\ast)
#       + \frac{1}{2} \gamma^2 L M
#     \end{equation*}
#   also
#     \begin{equation*} 
#     \mathbb{E}(f_{t+1}- f_\ast) 
#     \leq
#       (1 -\gamma\mu c) \mathbb{E}(f_t - f_\ast)
#       + \frac{1}{2} \gamma^2 L M
#     \end{equation*}
#   
# - subtrahiert man $\frac{\gamma L M}{2 \mu c}$
#   auf beiden Seiten, so ergibt sich
#     \begin{align*}
#     \mathbb{E}(f_{t+1}- f_\ast) - \frac{\gamma L M}{2 \mu c}
#     & \leq
#      (1 -\gamma\mu c) \mathbb{E}(f_t - f_\ast)
#      + \frac{1}{2} \gamma^2 L M - \frac{\gamma L M}{2 \mu c} \\
#     & = 
#      (1 -\gamma\mu c) 
#      \big(
#      \mathbb{E}(f_t - f_\ast) - \frac{\gamma L M}{2 \mu c}
#      \big)
#    \end{align*}
#    
# - wegen $\mu \leq L$ und $M_G \geq c_G^2 \geq c^2$ ist 
#     \begin{equation*} 
#     0 < \gamma\mu c 
#       \leq \frac{\mu c^2}{L M_G}
#       \leq \frac{c^2}{M_G}
#       \leq 1
#     \end{equation*}
#    
# $\square$

# Ohne "Rauschen" ist $M=0$ und
#   \begin{equation*} 
#   \mathbb{E}(f_{t} - f_\ast) 
#   \leq 
#   (1-\gamma \mu c)^{t-1} \mathbb{E}(f_1 - f_\ast)
#   \end{equation*}
# so dass wir eine Komplexität von $\mathcal{O}\big(\log(\frac{1}{\varepsilon})\big)$
#   erhalten.
#   
# Mt "Rauschen" haben wir
#   \begin{equation*} 
#   \mathbb{E}(f_{t} - f_\ast) 
#   \leq 
#   \underbrace{\frac{\gamma L M}{2 \mu c}}_{\text{fix}}
#   + \underbrace{
#     (1-\gamma \mu c)^{t-1} 
#     \big(\mathbb{E}(f_1 - f_\ast) - \frac{\gamma L M}{2 \mu c}\big)
#     }_{\text{geometrische Reduktion}},
#   \end{equation*}
# d.h. wir erreichen
#   \begin{equation*} 
#   \mathbb{E}(f_{t} - f_\ast) 
#   \leq \frac{\gamma L M}{2 \mu c} + \varepsilon
#   \end{equation*}
# in 
#   \begin{equation*} 
#   t = \mathcal{O}\big(\log(\frac{1}{\varepsilon})\big)
#   \end{equation*}
# Schritten.
#   

# $\frac{\gamma L M}{2 \mu c}$ wird klein, wenn $\gamma$ klein wird, wobei
#   dann aber $1 - \gamma\mu c$ nahe bei $1$ liegt, so dass die geometrische
#   Reduktion nur sehr langsam ist.
#   
# Dies führt auf die Idee des Restarts:
# 
#  - starte mit einer Schrittweite $\gamma^{(1)}$ und iteriere bis
#    \begin{equation*} 
#    \mathbb{E}(f_{t} - f_\ast) 
#    \leq \frac{\gamma^{(1)} L M}{2 \mu c} + \varepsilon^{(1)}
#    \end{equation*}
#   
#  - verkürze die Schrittweite auf $\gamma^{(2)} < \gamma^{(1)}$ und iteriere bis
#    die rechte Seite hinreichend klein ist
#    
# Das kann man soweit ausbauen, dass man in jedem Schritt die Schrittweite
#   $\gamma$ anpasst.

# **Satz:**
# Es sei $f$ differenzierbar, $L$-glatt, $\mu$-konvex. 
# Es existiere ein $x_\ast$ mit $f_\ast = f(x_\ast)=\inf_x f(x)$ und
# es sei
#   \begin{equation*} 
#   \gamma_t = \frac{\beta}{\gamma + t},
#   \quad
#   \beta > \frac{1}{\mu c},
#   \quad
#   \gamma > 0,
#   \quad
#   \gamma_1 < \frac{c}{L M_G}.
#   \end{equation*}
# Dann gilt
#   \begin{equation*} 
#   \mathbb{E}(f_{t} - f_\ast) 
#   \leq 
#   \frac{\nu}{\gamma + t}
#   \end{equation*}
# mit
#   \begin{equation*}
#   \nu = \max
#   \Big(
#     \frac{\beta^2 L M}{2(\beta\mu c -1)},
#     (\gamma+1)\ \mathbb{E}(f_{1} - f_\ast)
#   \Big).
#   \end{equation*}

# **Beweis:**
# 
# - aus den Voraussetzungen folgt
#    \begin{equation*} 
#    \gamma_t L M_G \leq \gamma_1 L M_G \leq c
#    \quad
#    \forall t
#    \end{equation*}
#  
# - mit $\gamma = \gamma_t$ folgt aus dem letzten Lemma
#    \begin{align*}
#    \mathbb{E}_{i_t}(f_{t+1}) - f_t 
#    &\leq
#      -\gamma_t \big(c - \frac{\gamma_t L}{2} M_G \big) \| f'_t \|_2^2 
#      + \frac{1}{2} \gamma_t^2 L M \\
#    &=
#      \big( -\gamma_t c + \gamma_t \frac{\gamma_t L  M_G}{2} \big) \| f'_t \|_2^2 
#      + \frac{1}{2} \gamma_t^2 L M \\
#    &\leq
#      - \frac{1}{2} \gamma_t c  \| f'_t \|_2^2   
#      + \frac{1}{2} \gamma_t^2 L M \\
#    \end{align*}
#      
# - wegen der $\mu$-Konvexität von $f$ auf $\mathbb{R}^d$ gilt wieder
#    \begin{equation*} 
#    \|  f'_t \|_2^2  \geq 2\mu(f_t - f_\ast)
#    \end{equation*}
#   und somit
#    \begin{equation*} 
#    \mathbb{E}_{i_t}(f_{t+1}) - f_t 
#    \leq
#      - \gamma_t \mu c  (f_t - f_\ast)   
#      + \frac{1}{2} \gamma_t^2 L M ,
#    \end{equation*}
#   bzw.
#    \begin{equation*} 
#    \mathbb{E}_{i_t}(f_{t+1}) - f_\ast - (f_t - f_\ast)
#    \leq
#      - \gamma_t \mu c  (f_t - f_\ast)   
#      + \frac{1}{2} \gamma_t^2 L M
#    \end{equation*}
#    
# - nun bilden wir auf beiden Seiten die Erwartungswerte 
#    $\mathbb{E}_{i_1} \ldots \mathbb{E}_{i_{t-1}}$ und erhalten
#    \begin{equation*} 
#    \mathbb{E}(f_{t+1} - f_\ast)
#    \leq
#      (1- \gamma_t \mu c) \ \mathbb{E}(f_t - f_\ast)   
#      + \frac{1}{2} \gamma_t^2 L M   
#    \end{equation*}
#    
# - per Induktion können wir nun die Aussage des Satzes beweisen
#  
# - für $t=1$ folgt die Behauptung direkt aus der Definition von $\nu$
#  
# - für den Schritt $t \to t+1$ erhalten wir
#    \begin{align*}
#    \mathbb{E}(f_{t+1} - f_\ast)
#    & \leq  
#      (1- \gamma_t \mu c) \ \mathbb{E}(f_t - f_\ast)   
#      + \frac{1}{2} \gamma_t^2 L M \\
#    & \leq  
#      \big(1- \frac{\beta}{\gamma + t}\mu c\big) \ \mathbb{E}(f_t - f_\ast)   
#      + \frac{1}{2} \frac{\beta^2}{(\gamma + t)^2} L M \\
#    & \leq  
#      \big(1- \frac{\beta}{\gamma + t}\mu c\big) \ \frac{\nu}{\gamma + t}   
#      + \frac{1}{2} \frac{\beta^2}{(\gamma + t)^2} L M \\
#    & =  
#      \frac{(\gamma + t - \beta\mu c)\nu}{(\gamma + t)^2}
#      + \frac{\beta^2 L M}{2 (\gamma + t)^2}  \\
#    & =  
#      \frac{\gamma + t - 1}{(\gamma + t)^2}\nu
#      - \frac{\beta\mu c-1}{(\gamma + t)^2}\nu
#      + \frac{\beta^2 L M}{2 (\gamma + t)^2}
#      \\
#    & =  
#      \frac{\gamma + t - 1}{(\gamma + t)^2}\nu
#      +
#      %\underbrace{
#      \frac{\beta^2 L M - 2(\beta\mu c-1)\nu}{2 (\gamma + t)^2}
#      %}_{\leq 0 \ \text{nach Definition von}\ \nu} 
#    \end{align*}
#    
# - nach Definition von $\nu$ gilt
#    \begin{equation*} 
#    \nu \geq \frac{\beta^2 L M}{2(\beta\mu c -1)}
#    \end{equation*}
#   also
#    \begin{equation*} 
#    \beta^2 L M - 2(\beta\mu c-1)\nu \leq 0
#    \end{equation*}
#   und somit 
#    \begin{align*}
#    \mathbb{E}(f_{t+1} - f_\ast) 
#    &\leq \frac{\gamma + t - 1}{(\gamma + t)^2}\nu \\
#    &\leq \frac{(\gamma + t - 1)(\gamma + t + 1)}{(\gamma + t)^2}
#          \frac{\nu}{\gamma + t + 1} \\
#    &= \frac{(\gamma + t)^2 - 1}{(\gamma + t)^2}
#          \frac{\nu}{\gamma + t + 1} \\
#    &\leq \frac{\nu}{\gamma + t + 1}
#    \end{align*}
#    
# $\square$

# ## Zusammenfassung

# Ist $f: \mathbb{R}^d \to  \mathbb{R}$ differenzierbar, $L$-glatt und $\mu$-konvex 
#    mit $f_\ast = f(x_\ast) = \inf_x f(x)$ dann haben wir für Stochastic-Gradient-Descent
#    \begin{equation*} 
#    x_{t+1} = x_t - \gamma_t f'_{i_t}(x_t)
#    \end{equation*}
#    folgende Ergebnisse gezeigt:
#    
#    - ist $\gamma$ konstant, dann ist
#      \begin{equation*} 
#      \mathbb{E}(f_t) - f_\ast \leq c(\gamma) + \varepsilon
#      \end{equation*}
#      in $\mathcal{O}(\log\frac{1}{\varepsilon})$ Schritten
#      
#    - ist $\gamma \sim \frac{1}{t}$, dann ist
#      \begin{equation*} 
#      \mathbb{E}(f_t) - f_\ast \leq  \varepsilon
#      \end{equation*}
#      mit $t = \mathcal{O}(\frac{1}{\varepsilon})$
#      
# - Gradient-Descent liefert für $f$ $L$-glatt und $\mu$-konvex $, 0 < \gamma \leq \frac{1}{L}$
#     \begin{equation*} 
#     f_{t}-f_\ast \leq \varepsilon
#     \end{equation*}
#     in $t = \mathcal{O}\Big( \log\big(\frac{1}{\varepsilon}\big) \Big)$ Schritten
#     
# Bei Stochastic-Gradient-Descent benötigen wir also deutlich mehr Schritte,
#   wobei jeder einzelne Schritt wegen der vereinfachten Gradientenberechnung $f'_{i_t}$
#   sehr viel weniger Aufwand verursacht.
#      
# Proximal- bzw. Subgradienten-Verfahren können analog "umgebaut" werden.
