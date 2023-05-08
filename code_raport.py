import pandas as pd
import numpy as np
import seaborn as sns
import scipy.optimize as sc
import matplotlib.pyplot as plt
from tqdm import trange
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.optim as optim
import visdom

loss_viz = visdom.Visdom()


c = 3e8

def GetRefractiveIndex(ri_file):
    """_summary_  takes the data from the data base 
    and converts it to a torch tensors 

    Parameters
    ----------
    ri_file : _type_ name of csv file for silver it would be "Ag"

    Returns
    -------
    W : _type_ torch.tensor
        _description_ list of frequencies 
    n : _type_ torch.tensor
        _description_ list of values of the optical index
    """

    a = pd.read_csv("BDD/Materiaux/" + ri_file + ".csv")
    sepRealImag = a.index[(a.loc[:,["wl","n"]] == np.array(["wl","k"])).iloc[:,0]]
    csvReal = None
    csvImag = None
    if sepRealImag.shape[0] > 0:
        sepRealImag = sepRealImag[0]
        csvReal = a.iloc[:sepRealImag,:].reset_index(drop=True).to_numpy().astype(float)
        csvImag = a.iloc[(sepRealImag+1):,:].reset_index(drop=True).to_numpy().astype(float)
    else:
        csvReal = a.to_numpy().astype(float)


    def extend(X):
        return np.unique(X)


    def interpolate(X, M, imag=False):
        Y = np.zeros_like(X)
        for i in range(X.shape[0]):
            x = X[i]
            if x in M[:,0]:
                Y[i] = M[M[:,0]==x,1]
            else:
                id_inf = 0
                id_sup = M.shape[0]-1
                if (M[:,0]<x).sum()>0:
                    id_inf = np.nonzero((M[:,0]<x))[0][-1]
                if (M[:,0]>x).sum()>0:
                    id_sup = np.nonzero((M[:,0]>x))[0][0]

                x_inf, x_sup = M[id_inf,0], M[id_sup,0]
                y_inf, y_sup = M[id_inf,1], M[id_sup,1]
                if x_inf >= x:
                    x_inf = x
                    y_inf = np.exp(-0.1/x) if imag else 1+np.exp(-0.1/x)
                if x_sup < x:
                    x_sup = x
                    y_sup += x

                Y[i] = y_inf + (y_sup - y_inf)/(x_sup - x_inf)*(x - x_inf)
        return Y

    X = csvReal[:,0]
    if not (csvImag is None):
        X = np.hstack([X, csvImag[:,0]]).astype(float)
        X = np.unique(np.sort(X))
    X = extend(X)
    nR = interpolate(X, csvReal)
    nI = np.zeros_like(nR)
    if not (csvImag is None):
        nI = interpolate(X, csvImag, imag=True)

    nR, nI = np.flip(nR).copy(), np.flip(nI).copy()
    n = nR + 1j*nI

    W = np.sort(2 * np.pi * c / (X * 1e-6)) * 1e-15
    nPts = 70
    if W.shape[0] > nPts:
        s = W.shape[0]//nPts
        n = n[0:n.shape[0]:s]
        W = W[0:W.shape[0]:s]

    return torch.from_numpy(W), torch.from_numpy(n)


def FitMOSEM(W, n, outputFile, nPoles=7, nPolesIm=2, nIter=2500, lr=3.3e-2, q1=1., q2=1.):
    """_summary_ Uses a gradient descent algorithm in order to find 
    optimal parameters to model data using the model derived from the pole expansion

    Parameters
    ----------
    W :  torch.tensor
        List of frequencies we have data on
    n :  torch.tensor
        mesurements of n for frequencies in W
    outputFile : string
                File name that will be used to save the parameters and
                to find the data
    nPoles : int, optional
        number of poles not situated on the imaginary axis, by default 7
    nPolesIm : int, optional
               number of poles situated on the imaginary axis, by default 2
    nIter : int, optional
        Amount of iteration the algorithm will do before stoping and returning
        the parameters, by default 2500
    lr : float, optional
         learning rate used by the opitimiser, by default 3.3e-2
    q1 : float, optional
         parameter used to balance the importance of real and imaginary part of n, by default 1.
    q2 : float, optional
         parameter used to balance the importance of real and imaginary part of n, by default 1.

    Returns
    -------
    torch.tensor 
    returns all the  parameters found by the gradient descent algorithm
    """

    scatterColors = np.random.randint(0, 255, (nPoles+nPolesIm+1, 3))

    def H(w, HNR, w0n, s1n, Gn, s2n, w0im, Gim, Gi0):
        #Calculates values of epsilon_r predicted by the pole expansion model given a set of parameters
        w1 = 0
        w2 = 0
        if len(w.shape) == 0:
            w1 = torch.from_numpy(np.repeat(w, w0n.shape[0])) + 0j
            w2 = torch.from_numpy(np.repeat(w, w0im.shape[0])) + 0j
        else:
            w1 = w[..., None].repeat_interleave(w0n.shape[0], len(w.shape))
            w2 = w[..., None].repeat_interleave(w0im.shape[0], len(w.shape))

        return 1 + HNR + 1j*Gi0/w - \
               ((w0im**2/Gim)/(w2**2 + 1j*torch.abs(Gim)*w2)).sum(dim=len(w2.shape)-1) - \
               ((1j*s1n*Gn*w1 + s2n*w0n**2)/((w1**2 - w0n**2) + 1j*torch.abs(Gn)*w1)).sum(dim=len(w1.shape)-1)


    def L(HNR, w0n, s1n, Gn, s2n, w0im, Gim, Gi0, n_it):
        #This is the cost function it returns the quantity that the gradient descent algorithm will try to minimise
        l = torch.zeros(1)
        Hw = H(W, HNR, w0n, s1n, Gn, s2n, w0im, Gim, Gi0)

        pI_min = 0.1
        N = 1000

        l += 0.005*torch.mean(torch.abs((Hw - n ** 2) ** 2))
        l += 0.001*torch.mean(torch.abs(torch.sqrt(Hw) - n) ** 2)
        #The previous contributions corespond to the module of the error on n and epsilon
        A = torch.mean(torch.abs(torch.real((Hw - n ** 2) ** 2)))
        #A represents the error commited on the real part of epsilon
        B = torch.mean(q1 * torch.abs(torch.imag((Hw - n ** 2) ** 2)))
        #B represents the error commited on the imaginary part of epsilon, it is balanced with A by q1
        l += torch.sqrt(A + B)
        l += 0.1*torch.sum(torch.relu(-np.imag(Hw) + 0.005))
        # we also represents the fact that epsilon has to a 
        #positive imaginary part.
        l += 0.03*torch.mean(torch.abs(s1n*Gn))
        #in order to remain close to the drude loren
        return l

    #We initialise all parameters for the initial guess
    HNR = 0.1*(torch.rand(1) + 2)
    HNR.requires_grad_(True)
    w0n = torch.linspace(1e-3, 2*W[-1], nPoles)
    w0n.requires_grad_(True)
    s1n = 0.1*(torch.rand(nPoles) + 2)
    s1n.requires_grad_(True)
    Gn = 0.01*torch.linspace(1e-3, 8, nPoles)
    Gn.requires_grad_(True)
    s2n = 0.1*(torch.rand(nPoles) + 2)
    s2n.requires_grad_(True)
    w0im = torch.linspace(1e-3, 8, nPolesIm)
    w0im.requires_grad_(True)
    Gim = 0.01 * torch.linspace(1e-3, 8, nPolesIm)
    Gi0 = torch.tensor([0.01])
    Gi0.requires_grad_(True)

    #the optimiser is the one that does the gradient descent algorithm it is given all parameters that can be changed
    optimizer = optim.AdamW([HNR, w0n, s1n, Gn, s2n, w0im, Gim, Gi0], lr=lr)
    L_array = np.zeros(nIter)
    i = 0
    L0 = torch.zeros(1)
    while i < nIter:
        #This loop is the execution of the gradient descent algorithm one iteration at a time
        optimizer.zero_grad()

        Ln = L(HNR, w0n, s1n, Gn, s2n, w0im, Gim, Gi0, i)
        dL = (Ln - L0).detach().numpy()
        L_array[i] = Ln.detach().numpy()
        if not (np.isnan(dL)):
            L0 = Ln

            Ln.backward()
            optimizer.step()

            i+=1

            vizDict = dict(xlabel="iteration",
                           ylabel="loss",
                           title="Cout (" + rf_file + ")",
                           ytype="log")
            loss_viz.line(X=np.arange(i),
                          Y=L_array[:i],
                          opts=vizDict,
                          win="cost")


            if (i%20) == 0:
            # every 20 iterations the values of epsilon are sent to a visual interface called visdom
                Hw = H(W, HNR, w0n, s1n, Gn, s2n, w0im, Gim, Gi0).detach().numpy()

                # REAL, IMAG, ABS CURVES
                Y0 = torch.real(n**2)
                Y1 = np.real(Hw)
                y = np.vstack([Y0, Y1])
                x = np.vstack([W, W])
                vizDict = dict(xlabel="frequence",
                               ylabel="H",
                               title="real",
                               xtype="log",
                               legend=["Exact", "Fitted"])
                loss_viz.line(X=x.transpose(),
                              Y=y.transpose(),
                              opts=vizDict,
                              win="curves_real")

                Y0 = torch.imag(n**2)
                Y1 = np.imag(Hw)
                y = np.vstack([Y0, Y1])
                x = np.vstack([W, W])
                vizDict = dict(xlabel="frequence",
                               ylabel="H",
                               title="imag",
                               xtype="log",
                               legend=["Exact", "Fitted"])
                loss_viz.line(X=x.transpose(),
                              Y=y.transpose(),
                              opts=vizDict,
                              win="curves_imag")

                Y0 = torch.abs(n**2)
                Y1 = np.abs(Hw)
                y = np.vstack([Y0, Y1])
                x = np.vstack([W, W])

                vizDict = dict(xlabel="frequence",
                               ylabel="H",
                               title="abs",
                               xtype="log",
                               legend=["Exact", "Fitted"])
                loss_viz.line(X=x.transpose(),
                              Y=y.transpose(),
                              opts=vizDict,
                              win="curves_abs")
        else:
            print("ERREUR")
            break
    #the next comand saves the parameters in the .npy format
    np.save("BDD/Materiaux/MOSEM/" + outputFile + "_HNR.npy", HNR.detach().numpy())
    np.save("BDD/Materiaux/MOSEM/" + outputFile + "_w0n.npy", w0n.detach().numpy())
    np.save("BDD/Materiaux/MOSEM/" + outputFile + "_s1n.npy", s1n.detach().numpy())
    np.save("BDD/Materiaux/MOSEM/" + outputFile + "_Gn.npy", Gn.detach().numpy())
    np.save("BDD/Materiaux/MOSEM/" + outputFile + "_s2n.npy", s2n.detach().numpy())
    np.save("BDD/Materiaux/MOSEM/" + outputFile + "_w0im.npy", w0im.detach().numpy())
    np.save("BDD/Materiaux/MOSEM/" + outputFile + "_Gim.npy", Gim.detach().numpy())
    np.save("BDD/Materiaux/MOSEM/" + outputFile + "_Gi0.npy", Gi0.detach().numpy())

    return HNR.detach().numpy(), w0n.detach().numpy(), s1n.detach().numpy(), Gn.detach().numpy(), s2n.detach().numpy(), w0im.detach().numpy(), Gim.detach().numpy(), Gi0.detach().numpy()

n_files = [ "Au2", "TiO2", "SiO2", "HfO2", "Co", "Ag", "Co", "Ta2O5"]
outputObj = []
for rf_file in n_files:
    W, n = GetRefractiveIndex(rf_file)
    nR = torch.real(n)
    nI = torch.imag(n)
    q1 = torch.abs(torch.real(n ** 2)) / torch.abs(torch.imag(n ** 2))
    q1[torch.abs(torch.imag(n ** 2)) < 1e-1] = 1
    q2 = torch.abs(nR) / torch.max(torch.abs(nI))
    q2[torch.abs(torch.imag(n)) < 1e-1] = 1
    q1[q1 <0.4] = 1
    q2[q2 <0.4] = 1
    
    outputObj.append(FitMOSEM(W, n, rf_file, nPoles=5, nPolesIm=3, nIter=10000, lr=0.05, q1=q1, q2=q2))

