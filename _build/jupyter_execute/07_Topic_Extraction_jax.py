#!/usr/bin/env python
# coding: utf-8

# # Topic Extraction, NMF

# ## Überblick

# Wir betrachten Low-Rank-Matrix-Approximationen. Mit deren Hilfe versuchen wir aus einer Menge von Texten "Themen" zu extrahieren ("Topic Extraction").

# ## Grundlagen

# Wir betrachten eine Menge von Texten und stellen diese zunächst in Form einer Matrix $V$ dar. Für jeden Text legen wir eine Spalte an, für jedes Wort eine Zeile und $v_{ij}$
# enthält dann die (skalierte) Häufigkeit, wie oft das $i$-te Wort im $j$-ten Text auftaucht.
# 
# Um wiederkehrende Strukturen (Cluster) in den Texten zu identifizeren kann man nun versuchen, die Matrix $V \in \mathbb{R}^{m \times n}$ zu zerlegen in
# \begin{equation*} 
# V = WH,
# \quad
# W \in \mathbb{R}^{m \times k},
# \quad
# H \in \mathbb{R}^{k \times n},
# \quad
# k \le \min(m,n).
# \end{equation*}
# Wenn $k\ll \min(m,n)$, dann können wir die Texte in $V$ mit Hilfe
# der "Basistexte" in den Spalten von $W$ erzeugen. Die Spalten von $W$
# stellen dann bestimmte "Topics" dar, die Spalten von $H$ geben an, wie
# diese Topics linear kombiniert werden, um den jeweiligen Ausgangstext zu erzeugen.

# In den seltensten Fällen wird eine exakte Zerlegung $V=WH$ möglich sein. Deshalb gibt man sich mit einer approximativen Zerlegung zufrieden, d.h. man sucht
# \begin{equation*} 
# \text{argmin}_{W,H}\|V-WH\|, 
# \end{equation*}
# \begin{equation*} 
# W \in \mathbb{R}^{m \times k},
# \quad
# H \in \mathbb{R}^{k \times n},
# \quad
# k \le \min(m,n).
# \end{equation*}

# Je nachdem welche Matrixnorm dabei verwendet wird, erhält man unterschiedliche Ergebnisse.
# Für $\|\cdot\|_2$ und $\|\cdot\|_\text{Fro}$ zeigt man, dass die Lösung dieses Problem durch TSVD gegeben ist
# (Satz von Eckart-Young-Mirsky), d.h. ist
# \begin{equation*} 
# V = \tilde{U}\tilde{\Sigma} \tilde{V}^T
# \end{equation*}
# eine SVD von $V$ und
# \begin{equation*} 
# \tilde{U}_k\tilde{\Sigma}_k \tilde{V}_k^T
# \end{equation*}
# eine TSVD,
# dann löst
# \begin{equation*} 
# W = \tilde{U}_k\tilde{\Sigma}_k,
# \quad
# H = \tilde{V}_k^T
# \end{equation*}
# bzw.
# \begin{equation*} 
# W = \tilde{U}_k,
# \quad
# H = \tilde{\Sigma}_k \tilde{V}_k^T
# \end{equation*}
# das Matrix-Approximationsproblem.

# Dieser Zugang hat einige Nachteile:
# 
# - die Lösung ist offensichtlich nicht eindeutig
# 
# - die Matrix $V$ ist in der Regel sparse, die TSVD nicht mehr
# 
# - für die Matrix $V$ gilt $v_{ij}\ge 0$, für $W,H$ aus TSVD gilt dies in der Regel
#   nicht mehr, was die Interpretation der Ergebnisse erschwert

# ## NMF

# Aus den oben genannten Gründen sucht man deshalb
# für $k \le \min(m,n)$
# eine Rang-$k$-Zerlegung von $V$ in nichtnegative Faktoren $W,H$,
# die sogenannte *Nonnegative-Matrix-Factorisation* (NMF),
# d.h.
# \begin{equation*} 
# \text{argmin}_{W,H}\|V-WH\|, 
# \end{equation*}
# \begin{equation*} 
# W \in \mathbb{R}^{m \times k},
# \quad
# H \in \mathbb{R}^{k \times n},
# \quad
# w_{ij}, h_{ij}\ge 0,
# \end{equation*}
# Dieses Problem ist nach wie vor nicht eindeutig lösbar
# ($WH = WA^{-1}AH$), außerdem gibt es dafür keinen Zugang
# aus der linearen Algebra, der analog zur TSVD für bestimmte Normen
# eine Lösung liefert. Allerdings kann man dafür sorgen, dass
# man die Sparsity der Matrizen beibehält.
# 
# Approximationen werden iterativ berechnet. 
# An dem folgenden Beispiel werden wir unterschiedliche Zugänge ausprobieren.

# ## Beispiel

# `sklearn` enthält den Datensatz `fetch_20newsgroups`, eine Sammlung von Newsgroup-Artikeln
# zu verschiedenen Themenbereichen.
# Wir laden Artikel aus mehreren Teilbereichen und versuchen daraus (ohne weitere
# Zusatzinformation, also "unsupervised") "Themen" zu extrahieren, d.h.
# Gruppen von Wörter, die in der Regel zusammen in einem Text auftauchen.
# 
# Wir laden zunächst die Daten. 

# In[1]:


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

seed = 17

from IPython.display import Math
get_ipython().run_line_magic('precision', '5')
np.set_printoptions(precision=5)


from sklearn.datasets import fetch_20newsgroups

categories = ['rec.autos', 'sci.space','sci.med']
remove = ('headers', 'footers', 'quotes')

train = fetch_20newsgroups(subset='train', remove=remove, categories=categories)
#test  = fetch_20newsgroups(subset='test' , remove=remove, categories=categories)

print("\n\n***************\n".join(train.data[:5]))


# Die zugehörigen Themengruppen sind

# In[2]:


np.array(train.target_names)[train.target[:5]]


# Jetzt erstellen wir eine Term-Document-Matrix $V$. $v_{ij}$ ist eine (skalierte Variante)
# der relativen Häufigkeit des Wortes $i$ im Dokument $j$

# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer

max_f = 2000
tv = TfidfVectorizer(max_df=0.95, min_df=2, max_features=max_f, stop_words='english')

V = tv.fit_transform(train.data).T

print(' Typ = {}\n Shape = {}\n Nonzeros = {}'.format(type(V), V.shape, V.nnz))

plt.spy(V, marker = '.', markersize = .1);
plt.axis('auto');


# Wir speichern uns noch die Wortliste zwischen.

# In[4]:


voc = np.array(tv.get_feature_names())
voc[-10:]


# Wir berechnen nun NMFs mit Parameter

# In[5]:


k = 10
print('k = {}'.format(k))


# ## NMF über Gradient-Descent

# Ist $A*B$ die komponentenweise Multiplikation der Matrizen $A$, $B$, dann minimieren wir
# die Zielfunktion
# \begin{equation*} 
# l(E,F) = \|V-(E*E)(F*F)\|_\text{Fro}^2. 
# \end{equation*}
# und setzen
# \begin{equation*} 
# W = E*E, \quad H=F*F
# \end{equation*}
# $l$ ist differenzierbar, die Nebenbedingung (Nicht-Negativität von $W,H$) ist in $l$ "eingebaut".

# In[6]:


# GPU: XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda jupyter-notebook
import jax
import jax.numpy as jnp

VV = np.array(V.toarray())

def l(E,F):
    return jnp.linalg.norm(VV - jnp.dot(E*E, F*F), ord='fro')**2

lE = jax.jacobian(l, 0)
lF = jax.jacobian(l, 1)

# Startwert erzeugen und vernünftig skalieren
m,n = VV.shape

np.random.seed(seed)
E = np.random.rand(m,k)
F = np.random.rand(k,n)

V0 = (E*E).dot(F*F)
s = np.linalg.norm(VV, ord='fro') / np.linalg.norm(V0, ord='fro')
sEF = s**0.25
E *= sEF
F *= sEF

# Iteration
nit = 100
ga  = 1e-1

# Start-Loss
print("||V - W0 H0||_Fro = {}".format(np.sqrt(l(E,F))))

for it in range(nit):
    E -= ga * lE(E,F)
    F -= ga * lF(E,F)

# End-Loss
print("||V - Wn Hn||_Fro = {}".format(np.sqrt(l(E,F))))


# Vergleicht man den letzten Wert mit $\|V\|_\text{Fro}$

# In[7]:


np.linalg.norm(VV, ord='fro')


# so sieht man, dass die Approximation recht grob ist. Außerdem sind die Matrizen $W,H$
# nicht dünn besetzt

# In[8]:


W = E * E
Ws = sp.sparse.csc_matrix(W)
Ws.shape, Ws.nnz


# In[9]:


Hs = sp.sparse.csc_matrix(F * F)
Hs.shape, Hs.nnz


# Die "Topics" isolieren wir jetzt, indem wir pro Spalte von $W$ die am höchsten gewichteten Worte betrachten:

# In[10]:


def topwords(v, n=7):
    for k,vk in enumerate(v):
        print(k, " ".join(voc[np.argsort(vk)[-n:]]))
    
topwords(W.T)


# ## NMF über Projected-Gradient-Descent

# Wir minimieren die differenzierbare Zielfunktion
# \begin{equation*} 
# l(W,H) = \|V-WH\|_\text{Fro}^2
# \end{equation*}
# und benutzen für die Einhaltung der Nebenbedingung 
# die Projektion (bezüglich der Forbenius-Norm) auf die nichtnegativen Matrizen
# \begin{equation*} 
# \Pi(A) = \big( \tilde{a}_{ij} \big)_{ij},
# \quad 
# \tilde{a}_{ij} = 
# \begin{cases}
# a_{ij} & a_{ij} \geq 0\\
# 0 & a_{ij} < 0
# \end{cases}
# \end{equation*}

# In[11]:


def l(W,H):
    return jnp.linalg.norm(VV - jnp.dot(W,H), ord='fro')**2

lW = jax.jacobian(l, 0)
lH = jax.jacobian(l, 1)

# Startwert erzeugen und vernünftig skalieren
m,n = VV.shape

np.random.seed(17)
W = np.random.rand(m,k)
H = np.random.rand(k,n)

V0 = W.dot(H)
s = np.linalg.norm(VV, ord='fro') / np.linalg.norm(V0, ord='fro')
sWH = np.sqrt(s)
W *= sWH
H *= sWH

# Iteration
nit = 100
ga  = 1e-1

# Start-Loss
print("||V - W0 H0||_Fro = {}".format(np.sqrt(l(W,H))))

for it in range(nit):
    W -= ga * lW(W,H)
    #W[W<0] = 0
    W = jax.ops.index_update(W, W<0., 0.)
    H -= ga * lH(W,H)
    #H[H<0] = 0
    H = jax.ops.index_update(H, H<0., 0.)

# End-Loss
print("||V - Wn Hn||_Fro = {}".format(np.sqrt(l(W,H))))


# Auch hier ist die Approximation recht grob, aber $W$ und $H$ sind weniger dicht besetzt

# In[12]:


Ws = sp.sparse.csc_matrix(W)
Ws.shape, Ws.nnz


# In[13]:


Hs = sp.sparse.csc_matrix(H)
Hs.shape, Hs.nnz


# Als Topics erhalten wir

# In[14]:


topwords(W.T)


# ## NMF aus Scikit-Learn

# In `sklearn` wird dazu
# für $\alpha,\gamma\ge 0$ das regularisierte Problem
# \begin{align*} 
# \text{argmin}_{W,H}
# \Big( 
# &\frac{1}{2}\|V-WH\|_\text{Fro}^2 \\
# &+ \alpha \gamma 
#   \big(\sum_{i,j}{|w_{ij}|} + \sum_{i,j}{|h_{ij}|} \big) \\
# &+ \frac{1}{2} \alpha(1-\gamma) (\|W\|_\text{Fro}^2 + \|H\|_\text{Fro}^2 )
# \Big)
# \end{align*}
# mit
# \begin{equation*} 
# \
# W \in \mathbb{R}^{m \times k},
# \quad
# H \in \mathbb{R}^{k \times n},
# \quad
# w_{ij}, h_{ij}\ge 0,
# \quad
# k \le \min(m,n).
# \end{equation*}

# Der letzte Term entspricht der Regularisierung bei Ridge-Regression, der mittlere Term ist vergleichbar mit der Regularisierung bei Lasso und sorgt für sparsity bei $W,H$. Zur näherungsweisen Lösung werden speziell angepasste Abstiegsverfahren benutzt.

# Ein Blick auf die Normen zeigt, dass $WH$ eine sehr grobe Approximation von $V$ ist.

# In[15]:


from sklearn.decomposition import NMF

model = NMF(n_components=k, init='random', random_state=seed)

W = model.fit_transform(V)
H = model.components_

np.linalg.norm(V-W.dot(H), ord='fro')


# In[16]:


np.linalg.norm(V.toarray(), ord='fro')


# $W$ und $H$ sind weniger dicht besetzt

# In[17]:


Ws = sp.sparse.csc_matrix(W)
Ws.shape, Ws.nnz


# In[18]:


Hs = sp.sparse.csc_matrix(H)
Hs.shape, Hs.nnz


# Als Topics erhalten wir

# In[19]:


topwords(W.T)


# ## Zusammenfassung

# NMF erzeugt eine approximative Matrixzerlegung
# \begin{equation*} 
# V \approx WH, \quad W,H\geq 0.
# \end{equation*}
# Die Zerlegung ist nicht eindeutig und wird i.d.R. iterativ berechnet.
