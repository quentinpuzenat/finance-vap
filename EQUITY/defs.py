import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time

def maximum(a, b):
        if a >= b:
            return a
        else:
            return b 

class Equity:
    def __init__(self, K, So, r, sigma, dt, IC, brownian_dims=100000):
        self.K = K,
        self.So =So,
        self.r = r,
        self.sigma =sigma,
        self.dt =dt,
        self.IC = IC,
        self.brownian_dims=brownian_dims #if you want to change dims of brownian simulations

    # on génère les browniens
    def compute_brownian(self):
        start = time()
        dim = self.brownian_dims

        # matrice des browniens
        self.W = np.ones((1, dim)) * self.So
        for i in range(1, 90): #attention ici dans les 90j on compte le day 0 avec la valeur 100
            self.W = np.vstack([self.W, self.W[i - 1]*np.exp((self.r[0] - (self.sigma[0]**2)/2)*i/365 + self.sigma[0]*np.sqrt(i/365)*np.random.randn(1, dim))])

        # on a le saut de 5 an moins 180 jours (donc dt = 1645/365)
        self.W2 = self.W[i - 1]*np.exp((self.r[0] - (self.sigma[0]**2)/2)*1645/365 + self.sigma[0]*np.sqrt(1645/365)*np.random.randn(1, dim))

        for i in range(1, 90): #attention ici il ne manque que 89 jours à calculer et on compte à rebours
            self.W2 = np.vstack([self.W2, self.W[i - 1]*np.exp((self.r[0] - (self.sigma[0]**2)/2)*(5 - ((90 - i)/365)) + self.sigma[0]*np.sqrt(5 - ((90 - i)/365))*np.random.randn(1, dim))])
            
        print(f"Temps de calcul: {(time() - start):.3f} sec pour {90*dim} simulations.")

    def compute_price(self):
        self.mean_first = list(pd.DataFrame(self.W).mean())
        self.mean_last = list(pd.DataFrame(self.W2).mean())

        self.payoff = [maximum(((i / j )- self.K[0])*100, 0) for i,j in zip(self.mean_last, self.mean_first)]

        print(f"Prix: {pd.Series(self.payoff).mean()*100}")
    

    def compute_convergence(self):
        start = time()
        
        # calcul de la convergence et de l'écart-type
        convergence = pd.Series(self.payoff).expanding().mean() *100
        ecart_type = pd.Series(self.payoff).expanding().std() *100

        x = pd.Series([i for i in range(1, self.brownian_dims + 1)]) *100
        # calcul des intervalles de confiance pour vérifier la convergence
        ic_moins = convergence - self.IC*ecart_type.div(np.sqrt(x)) 
        ic_plus = convergence + self.IC*ecart_type.div(np.sqrt(x)) 

        # on affiche le temps de calcul
        # ainsi que le prix avec l'intervalle de confiance associé
        print(f"Temps de calcul: {(time() - start):.3f} sec. pour {90*self.brownian_dims} valeurs.")
        print(f"Prix: {pd.Series(self.payoff).mean()*100} +- {maximum(list(convergence)[-1] - list(ic_moins)[-1], list(ic_plus)[-1] - list(convergence)[-1])*100:.3f}")
        
        # on plot le graphe
        x = [i for i in range(1, self.brownian_dims + 1)]
        plt.plot(x, convergence)
        plt.plot(x, ic_moins)
        plt.plot(x, ic_plus)
        plt.show()