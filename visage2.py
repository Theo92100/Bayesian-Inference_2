
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
from PIL import Image
from scipy.stats import multivariate_normal
from numpy.random import random
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import os
import time as t




def multinomial_resample(weights):

    weights=weights.T
    cumulative_sum = np.cumsum(weights)
    #cumulative_sum[-1] = 1.  # avoid round-off errors: ensures sum is exactly one
    #print ( np.searchsorted(cumulative_sum, random(len(weights))))
    return np.searchsorted(cumulative_sum, random(len(weights)))


def lecture_image() :
    #définit le chemin d’accès à la séquence d’images
    SEQUENCE = "./Image2/"
    #charge le nom des images de la séquence
    filenames = [f for f in os.listdir(SEQUENCE) if not f.startswith('.') and f.endswith('.png')]
    filenames = sorted(filenames)
    T = len(filenames)
    #charge la premiere image dans ’im’
    tt = 0

    im=Image.open((str(SEQUENCE)+str(filenames[tt])))
    #print(im)
    plt.imshow(im)

    return(im,filenames,T,SEQUENCE)

def selectionner_zone() :
    #lecture_image()
    print('Cliquer 4 points dans l image pour definir la zone a suivre.') 
    zone = np.zeros([2,4])
 #   print(zone))
    compteur=0
    while(compteur != 4):
        res = plt.ginput(1)
        a=res[0]
        #print(type(a)))
        zone[0,compteur] = a[0]
        zone[1,compteur] = a[1]
        plt.plot(a[0],a[1],marker='X',color='red')
        compteur = compteur+1

    #print(zone)
    newzone = np.zeros([2,4])
    newzone[0, :] = np.sort(zone[0, :])
    newzone[1, :] = np.sort(zone[1, :])

    zoneAT = np.zeros([4])
    zoneAT[0] = newzone[0,0]
    zoneAT[1] = newzone[1,0]
    zoneAT[2] = newzone[0,3]-newzone[0,0]
    zoneAT[3] = newzone[1,3]-newzone[1,0]
    #affichage du rectangle
    #print(zoneAT)
    xy=(zoneAT[0],zoneAT[1])
    rect=ptch.Rectangle(xy,zoneAT[2],zoneAT[3],linewidth=3,edgecolor='red',facecolor='None')
    #plt.Rectangle(zoneAT[0:1],zoneAT[2],zoneAT[3])
    currentAxis = plt.gca()
    currentAxis.add_patch(rect)
    plt.show(block=False)
    return(zoneAT)


def rgb2ind(im,nb) :
    #nb = nombre de couleurs ou kmeans qui contient la carte de couleur de l'image de référence
    print(im)
    image=np.array(im,dtype=np.float64)/255
    w,h,d=original_shape=tuple(image.shape)
    image_array=np.reshape(image,(w*h,d))
    image_array_sample=shuffle(image_array,random_state=0)[:1000]
    print(image_array_sample.shape)
   # print(type(image_array))
    if type(nb)==int :
        kmeans=KMeans(n_clusters=nb,random_state=0).fit(image_array_sample)
    else :
        kmeans=nb

    labels=kmeans.predict(image_array)
    #print(labels)
    image=recreate_image(kmeans.cluster_centers_,labels,w,h)
    #print(image)
    return(Image.fromarray(image.astype('uint8')),kmeans)

def recreate_image(codebook,labels,w,h):
    d=codebook.shape[1]
    #image=np.zeros((w,h,d))
    image=np.zeros((w,h))
    label_idx=0
    for i in range(w):
        for j in range(h):
            #image[i][j]=codebook[labels[label_idx]]*255
            image[i][j]=labels[label_idx]
            #print(image[i][j])
            label_idx+=1

    return image



def calcul_histogramme(im,zoneAT,Nb):

  #  print(zoneAT)
    box=(zoneAT[0],zoneAT[1],zoneAT[0]+zoneAT[2],zoneAT[1]+zoneAT[3])
   # print(box)
    littleim = im.crop(box)
##    plt.imshow(littleim)
##    plt.show()
    new_im,kmeans= rgb2ind(littleim,Nb)
    histogramme=np.asarray(new_im.histogram())
##  print(histogramme)
    histogramme=histogramme/np.sum(histogramme)
  #  print(new_im)
    return (new_im,kmeans,histogramme)

N=100
Nb=20

ecart_type=np.sqrt(50)
lambda_im=60
c1=pow(300,0.5)
c2=pow(300,0.5)
c3=pow(2/100,0.5)
C=np.diag([c1,c2,c3])

[im,filenames,T,SEQUENCE]=lecture_image()
zoneAT=selectionner_zone()


new_im,kmeans,histo_ref=calcul_histogramme(im,zoneAT,Nb)

def calcul_de_vraissemblance(histo, histo_ref,zone):
    #On teste si la zone est bien partiellement dans l'image, dans le cas contraire on renverra une vraissemblance très faible pour que la particule correspondante soit annihilée lors du rééchantillonnage
    if zone[0] > 500 or zone[0] < - zone[2] or zone[1] > 500 or zone[1] < - zone[3]:
        return(0)

    return np.exp(-lambda_im*(1-((histo*histo_ref)**(1/2)).sum()))

# Pour intégrer le facteur d'échelle dans le code existant, nous devons modifier plusieurs parties du code.
# Je vais fournir une version modifiée de la fonction `filtrage_particulaire` pour inclure le paramètre d'échelle.

def filtrage_particulaire_avec_echelle():
    # Initialisation des états avec le facteur d'échelle X3
    X_0 = np.array(np.random.randn(N,3))  # Ajout d'une colonne pour l'échelle
    X_0[:,:2] *= np.sqrt(50)  # Initialisation des positions comme avant
    X_0[:,2] = 1 + np.random.randn(N)*np.sqrt(2/100)  # Initialisation de l'échelle avec une variance faible
    w = []
    for i in range(N):
        w += [multivariate_normal.pdf(X_0[i], mean=[zoneAT[0],zoneAT[1],1], cov=C)]
    w = np.array(w)
    w = w/w.sum()

    N_eff = np.zeros(T)
    x_est = np.zeros((T-1,3))  # Modification pour inclure l'échelle dans l'estimation
    x = np.zeros((T,N,3))  # Stocke les positions et échelles des particules
    z = np.zeros((T,N,3))  # Initialisation de z pour les positions et échelles
    x[0,:] = X_0
    z[0,:] = X_0 

    for t in range(1,T):
        im = Image.open((str(SEQUENCE)+str(filenames[t])))
        A = np.random.choice(range(N),N,p=w)
        reech = x[t-1,:][A]  # Rééchantillonnage
        # Ajout de bruit aux positions et à l'échelle
        reech[:,:2] += np.array(np.random.randn(N,2))*np.sqrt(50)
        reech[:,2] += np.random.randn(N)*np.sqrt(2/100)  # Ajout de bruit à l'échelle
        x[t,:] = reech
        z[t,:] = reech
        # Calcul des vraisemblances pour les nouvelles particules avec ajustement pour l'échelle
        w = [calcul_de_vraissemblance(histo_ref,calcul_histogramme(im,[x[t,i,0],x[t,i,1],zoneAT[2]*x[t,i,2],zoneAT[3]*x[t,i,2]],kmeans)[2],[x[t,i,0],x[t,i,1],zoneAT[2]*x[t,i,2],zoneAT[3]*x[t,i,2]]) for i in range(N)]
        w = np.array(w)
        w=w/w.sum()  # Normalisation des poids

        # Calcul de N_eff(t)
        N_eff[t] = 1 / (w**2).sum()

        for i in range(N):
            x_est[t-1,:] += x[t,i,:]*w[i]

    return(x_est,z,N_eff)  # Retourne l'estimation avec l'échelle incluse

# Remarque : Cette fonction n'est qu'un squelette pour montrer comment intégrer le facteur d'échelle.
# Des ajustements supplémentaires peuvent être nécessaires pour s'assurer qu'elle fonctionne correctement avec le reste du code,
# notamment la mise à jour des fonctions qui dépendent de la structure du vecteur d'état.


# Utilisez la fonction filtrage_particulaire modifiée pour exécuter le filtrage
x_est, z, N_eff = filtrage_particulaire_avec_echelle()  # C'est la trajectoire estimée

input("Press Enter to continue.")  # Attend que l'utilisateur appuie sur Entrée
plt.close('all')  # Ferme toutes les figures ouvertes
plt.ion()  # Active le mode interactif

for t in range(1, T):
    plt.figure()  # Crée une nouvelle figure
    im = Image.open(os.path.join(SEQUENCE, filenames[t]))
    plt.imshow(im)
    
    # Trace un rectangle rouge correspondant à l'estimation sur cette image
    xy = x_est[t-1]
    rect = ptch.Rectangle((xy[0], xy[1]), zoneAT[2], zoneAT[3], linewidth=3, edgecolor='red', facecolor='none')
    currentAxis = plt.gca()
    currentAxis.add_patch(rect)
    
    # Affiche en vert toutes les particules qui ont permis de déterminer l'estimation
    for i in range(N):
        plt.plot(z[t, i, 0], z[t, i, 1], marker='X', color='green')
    
    plt.draw()  # Met à jour le dessin
    plt.pause(2)  # Pause pour permettre l'affichage de l'image avant de continuer
    plt.close()  # Ferme la figure courante avant de passer à la suivante

    

#Tracer de N_eff en fonction des images

plt.plot([i for i in range(T)], N_eff, label='lambda = 10')
plt.xlabel('t')
plt.ylabel('N_eff')
plt.legend()
plt.show()