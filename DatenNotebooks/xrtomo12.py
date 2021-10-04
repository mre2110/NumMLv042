# -*- coding: utf-8 -*-
"""
@author: mre

ohne Pandas

absolute Drehwinkel bei Strahlen

Animation mit Sinogramm, Deprecation Warning beseitigt

Python 3

Strahlen mit normalverteilter Störung (wegen Kantenproblematik)
"""
from pylab import *
import scipy.sparse as spa

from scipy.interpolate import interp1d
from numpy.random import seed

from matplotlib import animation, gridspec

#import pandas as pd


## Klassen
class gitter:
    def __init__(self, n):
        self.nx = n
        self.ny = n
        self.nc = self.nx * self.ny
        
        # Leitfähigkeit
        self.sigma = zeros((self.nx, self.ny))
        
        # Schnittpunkte mit Gitterlinien x=const
        self.hx = 1.0/self.nx
        self.hy = 1.0/self.ny
        
        self.xx = linspace(-0.5, 0.5, self.nx + 1)
        self.yy = linspace(-0.5, 0.5, self.ny + 1)
        
        # linke untere Ecke aller Rechtecke
        self.q0x, self.q0y = meshgrid(self.xx[:-1], self.yy[:-1])
        self.q0 = c_[self.q0x.flatten(), self.q0y.flatten()]
    
    def set_sigma(self, s = lambda x,y : 1.0 + (abs(x) < 0.3) * (abs(y)<0.2)):
        x = self.q0x + self.hx * 0.5
        y = self.q0y + self.hy * 0.5
        
        self.sigma[:] = s(x, y)

    def set_sigma1d(self, s):

        self.sigma.ravel()[:] = s
        
    def plot(self):
        # Kanten, eigentlich unnötig
#        xmin = min(self.xx)
#        xmax = max(self.xx)
#        
#        ymin = min(self.yy)
#        ymax = max(self.yy)
#        
#        plot(c_[self.xx, self.xx].T, c_[repeat(ymin, len(self.xx)), repeat(ymax, len(self.xx))].T, 'k')
#        plot(c_[repeat(xmin, len(self.yy)), repeat(xmax, len(self.yy))].T, c_[self.yy, self.yy].T, 'k')
        
        # Nummern
        #for i,xy in enumerate(self.q0):
        #    text(xy[0] + self.hx * 0.5 , xy[1] + self.hy * 0.5, str(i))
        
        # sigmafrom matplotlib import animation, gridspec
        x,y = meshgrid(self.xx, self.yy)
        pcolor(x, y, self.sigma, edgecolors='w')
        #colorbar()
        #axis('equal')


class strahl:
    # Winkel bei Drehung immer absolut
    # für Plots
    lmax = 2.5
    lmin = 0.0
         
    def __init__(self): 
        # Anzahl Strahlen
        self.ns = 1

        # Aufpunkt und Normalen,  Format n x 2
        self.g00 = zeros((self.ns, 2))
        self.gn0 = ones((self.ns, 2))
        
        self.dreh()
        # Richtungsvektor = Normale um 90 Grad gedreht      
        #self.gr# = self.gn.dot(array([[0.0, 1.0],[-1.0, 0.0]]))

    def plot(self):
        # Strahlen
        gmin = self.g0 + self.lmin * self.gr;
        gmax = self.g0 - self.lmax * self.gr;
        
        plot(c_[gmin[:,0], gmax[:,0]].T, c_[gmin[:,1], gmax[:,1]].T, 'y-')
        plot(gmin[:,0], gmin[:,1], 'yo')
        
        # Schirm
        ti = linspace(0, 1, self.ns)
        sx = interp1d(ti, gmax[:,0], kind = 'cubic')
        sy = interp1d(ti, gmax[:,1], kind = 'cubic')
        
        t = linspace(0,1)
        plot(sx(t), sy(t), 'b')
        #axis('equal')
    
    def plots(self, w, sc = lambda x : 1.0 - 0.1 * exp(-x*2)):
        # sc = sc if sc is not None else self.sc
        # plottet Daten w auf dem Schirm
        p = sc(w) * self.lmax
        #p = self.lmax * (1.0 - w * sc)
        gmax = self.g0 - c_[p, p] * self.gr
        
        ti = linspace(0, 1, self.ns)
        sx = interp1d(ti, gmax[:,0], kind = 'linear')
        sy = interp1d(ti, gmax[:,1], kind = 'linear')
        
        t = linspace(0,1)
        plot(sx(t), sy(t), 'r')
        
    def dreh(self, phi = 0.0):
        # Drehung immer bezüglich Anfangszustand
        p = phi * pi/180.0
        Q = array([[cos(p),sin(p)], [-sin(p),cos(p)]])
        
        self.g0 = self.g00.dot(Q.T)
        self.gn = self.gn0.dot(Q.T)
        
        self.gr = self.gn.dot(array([[0.0, 1.0],[-1.0, 0.0]]))


class parallel(strahl):
    #                  Anz.  AnzDreh Startw.   Endwinkel
    def __init__(self, ns=5, eps = 1e-6): 
        self.ns = ns
        # Winkel 0, Aufpunkt und Normale, Format n x 2
        self.lmax = 3.0
        self.g00 = c_[ (2*arange(1,ns+1)-1)/(2.0*ns) - 0.5, 1.5*ones(ns) ] + eps * randn(ns,2)
        self.gn0 = c_[ ones(ns)                           , zeros(ns)] + eps * randn(ns,2) 

        self.dreh()        
        # Richtungsvektor = Normale um 90 Grad gedreht      
        #self.gr = self.gn.dot(array([[0,1],[-1,0]]))


class zentral(strahl):
    #                  Anz.  Öffn.winkel AnzDreh Startw.   Endwinkel
    def __init__(self, ns=5, psi = 30.0, eps = 1e-6):      
        self.ns = ns
        self.lmax = 2.5
        # erst mal für Winkel 0, Aufpunkt und Normale, Format n x 2
        self.g00 = c_[ zeros(ns), ones(ns) ] + eps * randn(ns,2)
        
        psi  = psi * pi/180.0
        p    = linspace(-psi/2.0, psi/2.0, ns)
        self.gn0 = c_[cos(p), sin(p)] + eps * randn(ns,2)
        
        self.dreh()
        # Richtungsvektor = Normales = zentral(nstrahl) um 90 Grad gedreht      
        #self.gr = self.gn.dot(array([[0,1],[-1,0]]))
        
## Helfer
def qs(q0, a,b, g0,gn):
    # [l,u,v] = qs(q0,a,b, gn,g0)
    #
    # schneidet Rechteck q0+[0,a]x[0,b] die Gerade gn*(x-g0)=0
    #
    # wenn ja  : l Länge des Schnitts, u,v Schnittpunkte
    # wenn nein: l=0, u,v undef.
    #
    # q0, a, b : linke untere Ecke und Kantenlänge des Rechteck
    # gn, g0   : Normalenvektor und Aufpunkt der Geraden
    
    # definiere Kante als Gerade: x = k0 + l * k1
    # bestimme Schnittpunkt über
    #
    #  gn*(k0 + l * k1  - g0) = 0
    #
    # also
    #
    #  l = gn*(g0 - k0) / gn * kl
    #
    # wobei l in [0,1] liegen muss
    #
    # wird nichts gefunden, so ist l=0 und u,v auch (wegen sparse)
    
    # unten/oben/links/rechts
    ll = zeros(4)
    xs = zeros((4,2))
    
    #eps = 1e-6
            
    k0l = [q0             , q0 + array([0.0, b]), q0             , q0 + array([a, 0.0])]
    k1l = [array([a, 0.0]), array([a, 0.0]),      array([0.0, b]), array([0.0, b])]
    for i, (k0,k1) in enumerate(zip(k0l, k1l)):
        h = gn.dot(k1)
        if h != 0 :
            ll[i]  = gn.dot(g0 - k0) / h
        else:
            ll[i] = 17.0
            
        xs[i] = k0 + ll[i] * k1
    
    s, = where((0.0 <= ll) & (ll <= 1.0))
    ns = len(s)
    
    if ns > 1:
        u  = xs[s[0]]
        v  = xs[s[1:]]
        normuv2 = norm(v - u, 2, axis = 1)
        
        l = max(normuv2)
        i = argmax(normuv2)      
        v = v[i]
    else:
        l = 0.0
        u = array([0.0, 0.0])
        v = u

    return r_[l,u,v]

#qs(array([-0.4, -0.1]), 0.1, 0.1, array([-0.4, -1. ]), array([ -1.00000000e+00, -1.22464680e-16]))


def cut(c,s):
    A  = spa.lil_matrix((s.ns, c.nc))
    
    uv = zeros(4) # dummy für vstack
    
    for i in range(s.ns):
        def helper(q0):
            return qs(q0, c.hx, c.hy, s.g0[i], s.gn[i])
        
        w = array(list(map(helper, c.q0)))
        
        A[i,:] = spa.csr_matrix(w[:,0])
        
        ii = w[:, 0] > 0.0
        uv = vstack( (uv, w[ii, 1:]) )
        
    return A, uv[1:]
    

## Objekt
def sig0(x,y):
    w = 1.0 * ( 1.0 * (abs(x)<0.3) * (abs(y)<0.2)  - 1.0 * (abs(x)<0.1) * (-0.2<y) * (y<0) )
    return w

#def sig0(x,y):
#    w = 1.0 + 10.0 * (abs(x)<0.2) * (abs(y)<0.2)   
#    return w  


## Hauptfunktion
def tomo(sig0 = sig0, ngitter = 10, nstrahl = 5, winkel = arange(0, 360, 10), quelle = zentral, delta = 1e-2, fout = None):
    """  
    |
    | 
    sig0    : Funktion die für (x,y) den Leitfähigkeitswert liefert
    ngitter : Gitterauflösung (in beide Richtungen gleich) auf [-0.5,0.5] x [-0.5,0.5]
    nstrahl : Anzahl der Strahlen für Quelle
    winkel  : array der Drehwinkel
    quelle  : zentral oder parallel
    delta   : std des Messrauschens
    fout    : Daten werden als float32 in fout.csv.gz komprimiert geschrieben
    
    Rückgabewert: X, y, Animationsfunktion
        
        a(rep = False, interval = 200)
        
        rep      : Repeat
        interval : Zeitabstand zwischen Frames
    """
    
    ## Zufallszahlen wiederholbar machen
    seed(17)
    
    ## Messprogramm durchführen
    #  in A stehen spaltenweise die Impulsantworten
    #  jede Zeile von y enthält für einen Winkel die Intensitäten
    winkel = array(winkel)
    nwinkel = len(winkel)
    
    # Gitter erzeugen und Objekt setzen
    c = gitter(ngitter)
    c.set_sigma(sig0)
    
    # Strahlungsquelle festlegen
    s = quelle(nstrahl)
    
    
    # exakte Leitfähigkeitsverteilung
    xexakt = c.sigma.flatten()
    
    y = []
    for i,phi in enumerate(winkel):
        # Einzelmessung
        s.dreh(phi)
    
        Ai,_ = cut(c,s)
        
        if i==0:
            A,_ = cut(c,s)
        else:
            A  = spa.vstack( (A , Ai ) )
        
        yi = Ai.dot(xexakt)
        y.append(yi)
    
        #print(phi)
    
    
    ## Messfehler    
    y = array(y)
    yamax = abs(y).max()
    dy = yamax * delta * randn(*y.shape)
    y += dy
    
#    print(y.shape)
#    pcolor(y.T)
        
    #print(y.min(), y.max())
    

    ## Output mit Pandas falls fout <> None
    if fout != None:
#        wi = [ winkel[j] for j in range(nwinkel) for i in range(nstrahl) ]
#        st = [i for j in range(nwinkel) for i in range(nstrahl) ]
#        
#        Xy32 = pd.SparseDataFrame(A).assign(y = y.flatten(), Winkel = wi).astype(float32).assign(Strahl = st)        
#        
#        Xy32.columns = ['Pixel_{}'.format(i) for i in range(A.shape[1])] + ['y', 'Winkel', 'Strahl']
#        
#        Xy32.to_dense().to_csv(fout+'.csv.gz', compression = 'gzip')
        spa.save_npz(fout + '_X.npz', A)
        np.save(fout + '_y.npy', y)
    
    ## Animation
    def anim():
        def a(rep = False, interval = 200):
            fig = figure(figsize = (8,10))
            #fig = gcf()
            gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1]) 
            ax1 = subplot(gs[0])
            ax2 = subplot(gs[1])
            
            #fig, (ax1, ax2) = subplots(1, 2, figsize=(br, h1+h2))
            
            # Tomograph
            lim1 = (-2, 2)
            ax1.set_aspect(1)
            
            # Sinogramm
            dw0 = (winkel[ 1] - winkel[ 0]) / 2
            dw1 = (winkel[-1] - winkel[-2]) / 2
            lim2x = (winkel.min()-dw0, winkel.max()+dw1)
            lim2y = (-0.5, s.ns+0.5)
            #ax2.set_aspect(h2 / br)
            
            ss, ww = meshgrid(arange(s.ns), winkel)
            
            sino = 1.0 / (1.0 + y)
            smin = sino.min()
            smax = sino.max()          


            def update(i):
                ## Tomograph
                ax1.figure.sca(ax1)
                cla()
                c.plot()
                s.dreh(winkel[i])
                s.plot()
                s.plots(y[i])
                xlim(*lim1)
                ylim(*lim1)
                axis('off')
                
                ## Sinogramm
                ax2.figure.sca(ax2)      
                pcolormesh(ww[:i+1, :].T, ss[:i+1, :].T, sino[:i+1, :].T, vmin = smin, vmax = smax, shading='auto')
                #contourf(ww, ss, log(si))
                xlim(*lim2x)
                ylim(*lim2y)
                axis('off')
                xlabel('Winkel')
            
            return animation.FuncAnimation(fig, update, 
                    frames = arange(len(winkel)),
                    interval=interval, repeat = rep)
    
        return a
    
    return(A.tocsc(), y.ravel(), anim())


# # Test für jupyter-notebook
# _,_,t = tomo()
# from IPython.display import HTML
# anim = HTML(t().to_jshtml())


#def CDnumba(w0, X, y, alpha = 1e-4, nit = 100):
#    m, n = X.shape
#    w = w0.copy()
#    
#    na = n * alpha
#    
#    xkxk = np.array(X.multiply(X).sum(axis=0)).ravel()
#    r = y - X.dot(w)
#    
#    for it in range(nit):
#        for k in range(n):
#            xk   = X[:,k].toarray().ravel()
#            wk   = w[k]
#            yk   = r + xk * wk
#            xkyk = xk.dot(yk)
#            
#            if   xkyk < -na:
#                 w[k] = (xkyk + na) / xkxk[k]
#            elif xkyk > na:
#                 w[k] = (xkyk - na) / xkxk[k]
#            else:
#                 w[k] = 0.0
#            
#            r = r + xk * (wk - w[k])
#    return(w)
#
#X,y,_ = tomo()
#w0 = np.ones(X.shape[1])
#w = CDnumba(w0, X.tocsr(), y, 1.0)