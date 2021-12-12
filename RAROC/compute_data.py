import pandas as pd
import numpy as np
from scipy.optimize import fsolve

class Data:

    # init function
    def __init__(self, montant, duree, note, Type, type_garantie, decote, nominal, pays, prix, corr, fi, beta, TSR):
        self.montant = montant
        self.duree = duree
        self.note = note
        self.Type = Type
        self.type_garantie = type_garantie
        self.decote = decote
        self.nominal = nominal
        self.pays = pays
        self.prix = prix
        self.corr = corr
        self.fi = fi
        self.beta = beta
        self.TSR = TSR
        self.couts = 0
        self.PNB = 0
        self.EL = 0
        self.UL = 0
        self.data = pd.DataFrame({
        "Données": ["Montant", "Durée", "Note", "Type", "Garantie", "Décôte", "Nominal", "Pays", "Prix"],
        "Valeurs": [self.montant, self.duree, self.note, self.Type, self.type_garantie, f"{self.decote*100}%", self.nominal, self.pays, f"{self.prix*100}%"]})

    # useful function in order to compute RAROC
    def compute_raroc(self, rating_df, type_df, country_df):
        # contrôle de gestion
        PNB = self.montant * self.prix
        couts = 0.3 * PNB
        GDR = self.nominal*(1-self.decote) # global depository receipt

        # risk management
        EAD = self.montant - GDR
        PD = rating_df[rating_df.PD == self.note][self.duree].values[0]
        LGD = type_df[type_df.Type == self.Type]["LGD"].values[0]
        rating_pays = country_df[country_df.Pays == self.pays]["Rating"].values[0]
        PD_pays = rating_df[rating_df.PD == rating_pays][self.duree].values[0]
        LGD_pays = country_df[country_df.Pays == self.pays]["LGD"].values[0]
        
        if PD < PD_pays:
            EAD_transfert = country_df[country_df.Pays == self.pays]["Taux_transfert"].values[0] * EAD
        else:
            EAD_transfert = 0

        EL = (EAD - EAD_transfert)*PD*LGD + EAD_transfert*PD_pays*LGD_pays
        UL = (EAD - EAD_transfert)*LGD*self.fi*self.beta*np.sqrt(self.corr*PD*(1-PD))+EAD_transfert*LGD_pays*self.fi*self.beta*np.sqrt(self.corr*PD_pays*(1-PD_pays))

        self.RAROC = (PNB - couts - EL) / UL + self.TSR

        self.couts = couts
        self.PNB = PNB
        self.EL = EL
        self.UL = UL

        self.risk_df = pd.DataFrame({
            "Risk Management": ["EAD", f"PD ({self.note})", "LGD", "EAD Transfert", f"PD Pays ({rating_pays})", "LGD Pays", "EL", "UL", "---", "RAROC"],
            "Valeurs": [EAD, f"{PD*100}%", f"{LGD*100}%", EAD_transfert, f"{PD_pays*100}%", f"{LGD_pays*100}%", EL, f"{UL:.2f}", "-", f"{self.RAROC*100:.2f}%"]
        })

        if self.RAROC <= 0:
            print("Attention, RAROC négatif avec ces réglages !")

        return self.RAROC, PNB
        
    def compute_SB(self, part_SB):
        self.PNB_SB = part_SB/(1-part_SB)*self.PNB
        self.couts_SB = 0.3 * self.PNB_SB

        self.RAROC_SB = (self.PNB_SB - self.couts_SB + self.PNB - self.couts - self.EL) / self.UL + self.TSR 
        self.part_SB = part_SB

        if self.RAROC_SB <= 0:
            print("Attention, RAROC Side Business négatif avec ces réglages !")

        return self.RAROC_SB

    def compute_price(self, wanted_RAROC):
        func = lambda wanted_price : wanted_RAROC - (((0.7*(self.part_SB/(1-self.part_SB))+ 0.7)*self.montant * wanted_price - self.EL) / self.UL + self.TSR)
        price_initial_guess = 0.5
        price_solution = fsolve(func, price_initial_guess)
        print(f"Le prix est de {price_solution[0]*100:.4f}% pour un RAROC Side Business de {wanted_RAROC*100}% ") 

    def summary(self):
        return pd.DataFrame({
            "Variables" : [
                "Analysis Horizon",
                "PNB",
                "Costs",
                "Credit EBIT",
                "Expected Loss",
                "Risk-Adjusted Credit EBIT",
                "",
                "Economic Capital (UL)",
                "",
                "Taxes",
                "RAROC",
                "",
                "PNB Side Business",
                "Costs Side Business",
                "RAROC Side Business"
            ],
            "Valeurs" : [
                self.duree,
                self.PNB,
                self.couts,
                self.PNB - self.couts,
                self.EL,
                self.PNB - self.couts - self.EL,
                "",
                self.UL,
                "",
                self.TSR,
                f"{self.RAROC*100:.2f}%",
                "",
                self.PNB_SB,
                self.couts_SB,
                f"{self.RAROC_SB*100:.2f}%"
            ],
        })
