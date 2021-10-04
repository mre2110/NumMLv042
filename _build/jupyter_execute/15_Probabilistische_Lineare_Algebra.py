#!/usr/bin/env python
# coding: utf-8

# # Probabilistische Lineare Algebra

# ## Grundlagen

# Aus der linearen Algebra sind viele Matrix-Zerlegungen bekannt, z.B. LU, QR, SVD.
# Wir betrachten hier die SVD, also
#   \begin{equation*} 
#   A = U\Sigma V^T
#   \end{equation*}
# mit $U,V$ orthonormal und $\Sigma = \mathrm{diag}_i(\sigma_i)$.
#   
# Hat $A$ den Rang $r$, dann gilt
#   \begin{align*}
#    \sigma_i &> 0 \quad i=1,\ldots,r \\
#    \sigma_i &= 0 \quad i=r+1,\ldots
#   \end{align*}
# und man erhält
#   \begin{equation*} 
#   \underbrace{A}_{\mathbb{R}^{m \times n}}=
#   \underbrace{U_r}_{\mathbb{R}^{m \times r}}
#   \ \underbrace{\Sigma_r}_{\mathbb{R}^{r \times r}}
#   \ \underbrace{V_r^T}_{\mathbb{R}^{r \times n}}
#   \end{equation*}
# mit $r \leq \min(m,n)$.
# Ist $r \ll \min(m,n)$, dann können Operationen mit $A$, wie z.B.
#   Matrix-Vektor-Produkte, über die Zerlegung mit deutlich weniger
#   Aufwand berechnet werden.
# 
# Die Spalten von $U_r$ bilden eine Orthonormalbasis des Bildraums
#   $R(A)$, d.h. sie enthalten die komplette Information über  $R(A)$.
# Gilt nun
#   \begin{equation*} 
#   \sigma_1 \geq \ldots \geq \sigma_k \geq \sigma_{k+1}\geq \ldots \geq \sigma_r,
#   \quad
#   \sigma_1 \gg \sigma_{k+1}
#   \end{equation*}
# dann hatten wir bei TSVD einfach $\sigma_{k+1} = \ldots = \sigma_r =0$ gesetzt
# und damit eine Matrix-Approximation durchgeführt
#   \begin{equation*} 
#   A = U\Sigma V^T \approx U_k \Sigma_k V_k^T.
#   \end{equation*}
# Die Spalten von $U_k$ sind eine Orthonormalbasis eines Unterraums von $R(A)$.
# Wegen $\sigma_1 \gg \sigma_{k+1}$ "hoffen" wir, das wir damit den "wesentlichen Teil" 
#   von $R(A)$ erwischt haben und damit $U_k \Sigma_k V_k^T$ eine gute Approximation von
#   $A$ ist.

# Diese Vorgehensweise lässt sich verallgemeinern:
# 
# - für $A\in \mathbb{R}^{m\times n}$ sucht man eine Rang-$k$-Approximation
#   \begin{equation*} 
#   A \approx QC, \quad Q \in \mathbb{R}^{m\times k}, \quad C \in \mathbb{R}^{k\times n}
#   \end{equation*}
#   
# - dazu konstruiert man zunächst einen Unterraum von $R(A)$ der möglichst kleine 
#   Dimension $k$ haben sollte, aber gleichzeitig die wesentlichen Abbildungseigenschaften 
#   von $A$ wiedergeben soll
#   
# - eine Orthonormalbasis dieses Unterraums legen wir als Spalten in der Matrix
#   \begin{equation*} 
#   Q\in\mathbb{R}^{m\times k}
#   \end{equation*}
#   ab und approximieren $A$ durch Orthogonalprojektion in diesen Unterraum, d.h.
#   \begin{equation*} 
#   A \approx QQ^T A =  QC = B, \quad C = Q^T A \in \mathbb{R}^{k\times n}
#   \end{equation*}
#   
# - mit $Q$ erzeugen wir also eine niedrig dimensionale 
#   Matrix $C = Q^TA$ so dass $QC$ die Ausgangsmatrix $A$
#   approximiert
#     
# - auf $C$ wenden wir dann Standard-Zerlegungen (LU, QR, SVD) 
#   an, was aufgrund der niedrigen Dimension wenig Aufwand verursacht,
#   und konstruieren daraus approximative Zerlegungen für 
#   die Ausgangs-Matrix $A$

# Wir gehen also zweistufig vor:
# 
#   - bestimme $Q$
#   
#   - berechne die Zerlegung von $C = Q^T A$ und darüber
#     eine Zerlegung von $B = QC$, was dann eine 
#     approximative Zerlegung von $A$ darstellt
#   
# Für den ersten Teil werden wir probabilistische Methoden benutzen.
# 
# 
# **Beispiel:** Approximative SVD für $A\in \mathbb{R}^{m \times n}$
# 
# - $Q_k\in \mathbb{R}^{m \times k}$ orthonormal sei gegeben, $k\leq\min(m,n)$
#   
# - ist $C_k = Q_k^TA \in \mathbb{R}^{k \times n}$, dann ist
#   \begin{equation*} 
#   B_k = Q_kC_k = Q_k Q_k^TA \approx A
#   \end{equation*}
#   eine Rang-$k$-Approximation von $A$
#   
# - wir berechnen die SVD von $C_k$ und erhalten
#   \begin{equation*} 
#   C_k = \tilde{U}_k \Sigma_k V_k^T,
#   \quad
#   \tilde{U}_k\in \mathbb{R}^{k \times k},
#   \quad
#   \Sigma_k\in \mathbb{R}^{k \times k},
#   \quad
#   V_k^T \in \mathbb{R}^{k \times n}
#   \end{equation*}
#   
# - mit $A\approx B_k = Q_k C_k$ folgt dann
#   \begin{equation*} 
#   A\approx Q_k C_k  = \underbrace{Q_k \tilde{U}_k}_{U_k} \Sigma_k V_k^T,
#   \end{equation*}
#   mit $U_k\in \mathbb{R}^{m \times k}$ orthonormal
#   
#   
# Die offene Frage ist jetzt, wie wir geeignete Matrizen $Q_k$ finden können.

# ## Probabilistische Verfahren

# In diesem Abschnitt betrachten wir die Konstruktion von $Q_k$.
# 
# Ab jetzt lassen wir den Index $k$ weg.
# Wir müssen folgendes Problem lösen:
# 
# - gegeben ist $A\in \mathbb{R}^{m \times n}$, $\varepsilon > 0$
#   
# - finde eine orthonormale Matrix $Q\in \mathbb{R}^{m \times k}$,
#     $k = k(\varepsilon) \leq \min(m,n)$, mit
#     \begin{equation*} 
#     \|A -QQ^TA\| \leq \varepsilon,
#     \end{equation*}
#   d.h.
#     \begin{equation*} 
#     \|A - X\| \leq \varepsilon
#     \end{equation*}
#   mit $X=QQ^TA$ mit $\mathrm{rang}(X) \leq k$
#     
# $k(\varepsilon)$ sollte so klein wie möglich sein.
# Als Matrix-Norm benutzen wir in der Regel
#   \begin{align*} 
#   \|A\|_2 
#   &= \sqrt{\rho(A^TA)},
#   \\
#   \|A\|_F 
#   &= \sqrt{\mathrm{tr}(A^TA)} = \sqrt{\sum_{i=1}^m\sum_{j=1}^n a_{ij}^2}.
#   \end{align*}
# Für orthonormal-invariante Matrix-Normen, d.h.
#   \begin{equation*} 
#   \|A\| = \|QA\| = \|A\tilde{Q}\|
#   \quad
#   \forall Q,\tilde{Q} \ \text{orthonormal}
#   \end{equation*}
# kennt man eine Optimal-Lösung dieses Problems.

# **Satz (Eckart, Young, Mirsky):** Ist $A\in \mathbb{R}^{m \times n}$ 
# mit SVD $A=U\Sigma V^T$ und die Matrixnorm $\|\cdot\|$ orthonormal invariant, dann
# gilt
#   \begin{equation*} 
#   \min_{X\in \mathbb{R}^{m \times n}, \mathrm{rang}(X)\leq k} \|A-X\| = \sigma_{k+1}.
#   \end{equation*}
# Das Minimum wird für
#   \begin{equation*} 
#   X_{\min} = U_k \Sigma_k V_k^T,
#   \end{equation*}
# also TSVD, angenommen.

# Die Normen $\|\cdot\|_2$ und $\|\cdot\|_F$ sind orthonormal 
# invariant (Übung), somit können wir das Ergebnis hier in anwenden.
# In unserem Set-Up bedeutet das:
# 
# - in $X_{\min}$ kommen nur die Spalten $u_1,\ldots,u_k$ von $U$ vor, d.h.
#     \begin{equation*} 
#     Q = U_k = (u_1,\ldots,u_k) \in \mathbb{R}^{m\times k}
#     \end{equation*}
#     
# - somit ist
#   \begin{align*}
#   QQ^TA 
#   &= U_k U_k^T U\Sigma V^T \\
#   &= U_k (I_k,0)\Sigma V^T \\
#   &= U_k (\Sigma_k,0) V^T \\
#   &= U_k \Sigma_k V_k^T 
#   \end{align*}
#   und wir könnten wie folgt vorgehen:
#   
#     - gebe $\varepsilon > 0$ vor
#     
#     - suche $k$ mit $\sigma_{k+1}<\varepsilon$
#     
#     - benutze $Q=U_k$ und 
#     \begin{equation*} 
#     B = QQ^TA = U_k \Sigma_k V_k^T 
#     \end{equation*}
#     als Rang-$k$-Approximation
#     
# Das Problem dabei ist, dass man dazu eine komplette SVD von $A$ benötigt.

# Wir müssen also überlegen, wie wir "billiger" an $Q$ kommen können.
# Dazu betrachten wir für $A\in \mathbb{R}^{m \times n}$ zwei Fälle:
# 
# - fixed-presicion: 
#   gegeben $\varepsilon > 0$, 
#   suche $Q$ orthonormal mit 
#   $\mathrm{rang}(Q)\leq k(\varepsilon)$,  
#   wobei
#   $k(\varepsilon)$ möglichst klein sein soll,
#   und 
#   \begin{equation*} 
#   \|A -QQ^TA\| \leq \varepsilon
#   \end{equation*}
#   
# - fixed-rank: $k,p$ ist gegeben, suche $Q \in \mathbb{R}^{m\times (k+p)}$ orthonormal mit 
#   $\mathrm{rang}(Q)\leq k$
#   und 
#   \begin{equation*} 
#   \|A -QQ^TA\| 
#   \approx
#   \min_{X\in \mathbb{R}^{m \times n}, \mathrm{rang}(X)\leq k} \|A-X\| 
#   = 
#   \sigma_{k+1}
#   \end{equation*}
#   
# Bei fixed-rank lassen wir bei $Q$ mehr Spalten zu als für eine Rang-$k$-Approximation 
#   unbedingt nötig wären ($p$).
# Dieser Freiheitsgrad wird später eine wichtige Rolle bei der Genauigkeit
#   probabilistischer Methoden spielen.
#   
# Wir untersuchen zunächst fixed-rank und knacken damit dann später auch fixed-precision.

# ## Fixed-Rank

# Wie kann der "Zufall" bei der Konstruktion von $Q$ helfen?
# 
# Wir betrachten dazu zunächst $A\in \mathbb{R}^{m \times n}$ mit $\mathrm{rang}(A)=k$:
# 
# - ziehe $k$ Zufallsvektoren $\omega^{(i)}$ (Verteilung zunächst uninteressant)
#   und berechne 
#   \begin{equation*} 
#   y^{(i)} = A \omega^{(i)}, \quad i=1,\ldots,k
#   \end{equation*}
#   
# - mit hoher Wahrscheinlichkeit gilt:
#   
#   - die $\omega^{(i)}$ sind linear unabhängig
#     
#   - $\omega^{(i)} \not\in N(A)$
#     
# - damit sind dann auch die $y^{(i)}$ mit hoher Wahrscheinlichkeit linear unabhängig
#   
# - wegen $\mathrm{rang}(A)=k$ gilt dann
#   \begin{equation*} 
#   R(A)=\mathrm{span}\big(y^{(1)},\ldots,y^{(k)}\big),
#   \end{equation*}
#   d.h. $Y = \big(y^{(1)},\ldots,y^{(k)}\big)$ spannt $R(A)$ auf
#   
# - orthonormalisieren wir jetzt $Y$ (modifizierter Gram-Schmidt, QR, etc.), so erhalten wir
#   $Q\in \mathbb{R}^{m \times k}$

# Jetzt sei $A = A_0 + E$, $\mathrm{rang}(A_0)=k$, wobei $E$ sei eine "kleine" Störung ist.
#   
#   - wir suchen einen Unterraum in $R(A)$, der einen möglichst "großen Teil" von
#     $R(A_0)$ abdeckt, aber nicht unedbingt die kleinstmögliche Dimension $k$ hat
#     sondern Dimension $k+p$
#     
#   - wir betrachten wieder
#     \begin{equation*} 
#     y^{(i)} = A \omega^{(i)} = A_0 \omega^{(i)} + E  \omega^{(i)}
#     \end{equation*}
#   
#   - der "Störanteil" $E  \omega^{(i)}$ "schiebt" $A_0 \omega^{(i)}$ eventuell
#     aus $R(A_0)$ hinaus
#   
#   - wenn wir genau $k$ verschiedene $y^{(i)}$ benutzen, dann ist die Wahrscheinlichkeit 
#     hoch, dass wir "wichtige Teile" von $R(A_0)$ nicht erfassen
#     
#   - deshalb benutzen wir $k+p$ Vektoren $y^{(i)}$, $p\in\mathbb{N}$

# Fassen wir diese Schritte zusammen, so erhalten wir das **Fixed-Rank-Verfahren**:
# 
#   - gegeben ist $A\in \mathbb{R}^{m \times n}$, $k,p\in \mathbb{N}$
#   
#   - bestimme eine "zufällige" Matrix $\Omega \in \mathbb{R}^{n \times (k+p)}$
#   
#   - berechne damit
#   \begin{equation*} 
#   Y = A\Omega \in \mathbb{R}^{m \times (k+p)}
#   \end{equation*}
#   
#   - orthonormiere die Spalten von $Y$ und erzeuge damit die orthonormale Matrix
#   \begin{equation*} 
#   Q \in \mathbb{R}^{m \times (k+p)}
#   \end{equation*}
#   
#   
# Dabei sind folgende Fragen noch offen:
# 
# - welches $\Omega$ und welches $p$ soll benutzt werden?
#   
# - wie orthonormalisiert man $Y$, das in der Regel schlechte Kondition hat
#   (Spalten sind "fast" linear abhängig)?
#   
# - wie ist der Aufwand, die Genauigkeit?
#   
# - wie geht fixed-prescision?

# **Satz:** $A \in \mathbb{R}^{m \times n}$, $k\geq 2$, $p\geq 2$, $k+p \leq \min(m,n)$,
# \begin{equation*} 
# \Omega = \big(\omega_{ij}\big)_{i,j} \in  \mathbb{R}^{n \times (k+p)},
# \quad
# \omega_{ij} \sim \mathcal{N}(0,1) \ \text{iid},  
# \end{equation*}
# dann gilt
# \begin{align*} 
# \mathbb{E}_\Omega\big(\|A - QQ^TA\|_2\big) 
# &\leq (1+\delta)\sigma_{k+1},
# \\
# \delta 
# &= \frac{4\sqrt{k+p}}{p-1}\sqrt{\min(m,n)}
# \end{align*}
# bzw.
# \begin{align*} 
# \mathbb{P}\big(\|A - QQ^TA\|_2 
# \leq (1+\tilde{\delta})\sigma_{k+1}\big)
# &\geq
# 1 - 3p^{-p},
# \\
# \tilde{\delta} 
# &= 9 \sqrt{k+p} \sqrt{\min(m,n)}.
# \end{align*}

# **Bemerkung:**
# 
# - $\min_{X\in \mathbb{R}^{m \times n}, \mathrm{rang}(X)\leq k} \|A-X\|_2 = \sigma_{k+1}$ 
#   so dass $\delta$, $\tilde{\delta}$ die Abweichung vom Optimum angibt
#     
# - $\delta$, $\tilde{\delta}$ wächst nur schwach in $k,m,n$
#   
# - es ist (unabhängig von $k,m,n$)
#   \begin{equation*} 
#   3 p^{-p}
#   \approx
#   \begin{cases}
#   10^{-3} & p=5 \\
#   3\cdot 10^{-10} & p=10 
#   \end{cases},
#   \end{equation*}
#   so dass für recht kleine $p$-Werte
#   $\mathbb{P}\big(\|A - QQ^TA\|_2 \leq (1+\tilde{\delta})\sigma_{k+1}\big)\approx 1$
#   
# - fallen die Singulärwerte schnell ab, so dass $\sigma_{k+1}$ klein ist, dann
#   sind die Approximationen sehr gut

# ## Randomized SVD

# Wir wenden jetzt die Methoden des letzten Abschnitts, um eine approximative SVD
#   von $A$ zu berechnen.
#   
# Eine direkte Anwendung liefert oft schlechte Ergebnisse, insbesondere wenn die 
#   Singulärwerte von $A$ nur langsam abfallen.
# Statt $Y=A\Omega$ benutzt man dann die Power-Methode 
#   \begin{equation*} 
#   Y = (AA^T)^q A\Omega,
#   \end{equation*}
# wobei in der Praxis meist $q=1$ oder $q=2$ ausreicht.
#   
# Es gilt $R\big((AA^T)^q A\big)\subset R(A)$ und mit $A=U\Sigma V^T$ folgt
#   \begin{equation*} 
#   AA^T = U\Sigma\Sigma^T U^T
#   \end{equation*}
#   also
#   \begin{align*} 
#   (AA^T)^q A 
#   &= (U\Sigma\Sigma^T U^T)^q U\Sigma V^T\\
#   &= U(\Sigma\Sigma^T)^q\Sigma V^T\\
#   &= U\Sigma_{q+1} V^T
#   \end{align*}
#   mit
#   \begin{equation*} 
#   \quad
#   \Sigma_{q+1} = \mathrm{diag}(\sigma_i^{q+1}),
#   \end{equation*}
#   wobei $\sigma_i^{q+1}$ für $q>0$ schneller abfällt als $\sigma_i$. 

# Damit erhalten wir schließlich folgenden Algorithmus für RSVD:
# 
# - $A \in \mathbb{R}^{m \times n}$, $k\in\mathbb{N}$, $p,q\in\mathbb{N}_0$
#   
# - Schritt 1:
#   
#   - $\Omega = \big(\omega_{ij}\big)_{i,j} \in \mathbb{R}^{n \times (k+p)}$,
#     $\omega_{ij} \sim \mathcal{N}(0,1)$ iid  
# 
#   - $Y = (AA^T)^q A\Omega \in \mathbb{R}^{m \times (k+p)}$
#     
#   - orthonormalisiere die Spalten von $Y$ und lege das Ergebnis in
#     $Q\in \mathbb{R}^{m \times (k+p)}$ ab
#   
# - Schritt 2:
#     
#   - $C = Q^T A\in \mathbb{R}^{(k+p) \times n}$
#     
#   - berechne SVD von $C$, $C = \tilde{U}\Sigma V^T$
#     
#   - setze $U = Q\tilde{U}$
#     
# - insgesamt ist dann $A \approx U\Sigma V^T$

# **Bemerkung:**
# 
# - man berechnet eine Rang-$(k+p)$-Approximation um eine (gute) Näherung für die
#   ersten $k$ Singulärwerte und -vektoren zu erhalten
#    
# - die Produkte $AA^TA\Omega$ sind anfällig gegen Rundungsfehler, weshalb man nach
#   jedem Schritt orthonormalisieren sollte
#    
#    
# **Satz:** $A \in \mathbb{R}^{m \times n}$, $q\in\mathbb{N}_0$, 
# $2 \leq k \leq \frac{1}{2}\min(m,n)$, dann gilt für $U,\Sigma,V$ aus der RSVD
# mit $p=k$
#   \begin{align*} 
#   \mathbb{E}_\Omega\big(\|A - U\Sigma V^T\|_2\big) 
#   &\leq 
#   (1+\delta)^{\frac{1}{2q+1}}\sigma_{k+1},
#   \\
#   \delta &= 4 \sqrt{\frac{2\min(m,n)}{k-1}}
#   \end{align*}  
# 
# 
# Da wir nur an den ersten $k$ Approximationen der Singulärwerte und -vektoren
# interessiert sind, ist es naheliegend, die $\sigma_i$ für $i>k$ abzuschneiden.
# Wir ersetzen oben $\Sigma$ durch 
# \begin{equation*} 
# \Sigma_k =  \mathrm{diag}(\sigma_1,\ldots,\sigma_k,0,\ldots,0),
# \end{equation*}
# d.h. wir benutzen
# eine TSVD von $C$ in Schritt 2.
# In diesem Fall ändert sich die Fehleranschätzung des letzten Satzes geringfügig.
# 
# 
# **Satz:** $A \in \mathbb{R}^{m \times n}$, $q\in\mathbb{N}_0$, 
# $2 \leq k \leq \frac{1}{2}\min(m,n)$, dann gilt für $U,\Sigma_k,V$ aus der RSVD
# mit $p=k$
# \begin{align*} 
# \mathbb{E}_\Omega\big(\|A - U\Sigma_k V^T\|_2\big) 
# &\leq 
# (1+\delta)^{\frac{1}{2q+1}}\sigma_{k+1} + \sigma_{k+1},
# \\
# \delta &= 4 \sqrt{\frac{2\min(m,n)}{k-1}}
# \end{align*}
#   
#   
# Damit ist die fixed-rank Variante der RSVD soweit erledigt.

# ## Fixed-Precision

# Das fixed-precision Problem könnte wie folgt auf die fixed-rank Variante
# zurückgeführt werden:
#   
# - starte mit fixed-rank mit kleinem $k$
#   
# - teste, ob $\|A - QQ^TA\|_2 \leq \varepsilon$
#   
# - wenn nicht, dann erhöhe $k$ und wiederhole den vorherigen Schritt
#   
#   
# Das Berechnen der Matrixnorm $\|A - QQ^TA\|_2$ ist in der Regel zu aufwendig,
# deshalb benötigt man dafür einen einfachen a-posteriori Fehlerindikator.
# Auch hier hilft uns wieder der "Zufall" weiter.
# 
# 
# **Lemma:** Sei $B\in\mathbb{R}^{m\times n}$, $\omega^{(j)}\in\mathbb{R}^n$,
# $j=1,\ldots,r$,
# $\omega_i^{(j)} \sim \mathcal{N}(0,1)$  iid und $\alpha > 1$.
# Dann gilt
# \begin{equation*} 
# \mathbb{P}
# \big(
# \|B\|_2 
# \leq 
# \alpha\sqrt{\frac{2}{\pi}} \ \max_{j=1,\ldots,r}\|B\omega^{(j)}\|_2
# \big)
# \geq
# 1 - \alpha^{-r}.
# \end{equation*}
#   

# Für die Praxis bedeutet das:
# 
# - erzeuge $\omega^{(j)}$,  $j=1,\ldots,r$ für $r$ klein
#   
# - mit $m_r = \max_{j=1,\ldots,r}\|(A-QQ^TA)\omega^{(j)}\|_2$ und $\alpha=10$ folgt
#   \begin{equation*} 
#   \mathbb{P}
#   \big(
#   \|A-QQ^TA\|_2 
#   \leq 
#   10\sqrt{\frac{2}{\pi}} \ m_r
#   \big)
#   \geq
#   1 - 10^{-r},
#   \end{equation*}
#   so dass $r\leq 10$ ausreichend ist

# Prinzipiell wenden wir beim diesem Fehlerschätzer die gleichen Operationen
# wie bei der Gaussian-Projection
# \begin{equation*} 
# Y = A\Omega,
# \quad
# \Omega = \big(\omega_{ij}\big)_{i,j} \in  \mathbb{R}^{n \times (k+p)},
# \quad
# \omega_{ij} \sim \mathcal{N}(0,1) \ \text{iid}
# \end{equation*}
# an, nur dass wir statt der Matrix $\Omega$ einzelne Vektoren $\omega^{(j)}$
# benutzen.
#   
# Ordnen wir die Berechnungsschritte etwas um, dann können wir den Fehlerschätzer
# praktisch ohne Mehraufwand in unser fixed-rank-Verfahren integrieren:

# - gegeben sind $A\in\mathbb{R}^{m\times n}$, $r\in\mathbb{N}$, $\varepsilon>0$:
# 
#   - erzeuge $\omega^{(j)}\in\mathbb{R}^n$, $j=1,\ldots,r$,
#     $\omega_i^{(j)} \sim \mathcal{N}(0,1)$  iid 
#     
#   - $y^{(j)} = A \omega^{(j)}$, $j=1,\ldots,r$
# 
#   - $k=0$, $Q^{(0)} = [\ ]$ (leer)
#   
#   - wiederhole bis 
#     $\displaystyle 
#     \max_{j=1,\ldots,r}\|y^{(k+j)}\|_2 \leq \frac{\varepsilon}{10\sqrt{\frac{2}{\pi}}}$:
#     
#     - $k := k+1$
#     
#     - $y^{(k)} := \big(I - Q^{(k-1)}{Q^{(k-1)}}^T\big) y^{(k)}$
#     
#     - $q^{(k)} := \frac{y^{(k)}}{\|y^{(k)}\|_2}$
#     
#     - $Q^{(k)} = [Q^{(k-1)}, q^{(k)}]$
#     
#     - erzeuge $\omega^{(k+r)} \in \mathbb{R}^n$, 
#       $\omega^{(k+r)} \sim \mathcal{N}(0,1)$ iid
#       
#     - $y^{(k+r)} := \big(I - Q^{(k)}{Q^{(k)}}^T\big) A \omega^{(k+r)}$
#     
#     - $y^{(l)} := y^{(l)} - {q^{(k)}}^T y^{(l)} {q^{(k)}}$ für $l = k+1,k+2,\ldots,k+r-1$
#     
#   - mit $Q = Q^{(j)}$ gilt dann 
#   \begin{equation*} 
#   \mathbb{P}\big(\|A-QQ^TA\|_2 \leq \varepsilon\big) \geq 1- 10^{-r}\min(m,n)
#   \end{equation*}
#     
# Damit ist $Q$ bestimmt.
# Der Rest der RSVD ist identisch zum fixed-rank-Fall, insbesondere
# kann dabei auch wieder die Power-Methode benutzt werden.

# ## Zusammenfassung

# Wir haben probabilistische Methoden für approximative Berechnung von TSVD
# betrachtet und Algortihmen für fixed-rank und fixed-precision Aufgabenstellungen
# vorgestellt.
