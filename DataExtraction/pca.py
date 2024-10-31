#%%
import sys 
import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits	
from matplotlib.pyplot import plot, ion, show
from mpl_toolkits.axes_grid1 import make_axes_locatable
from threadpoolctl import threadpool_limits



def plotSingle(components, eigenValues, scores, original_shape):
	for idx in range(0, components.shape[0]):
		fig, (ax1, ax2) = plt.subplots(1, 2)
		fig.suptitle("Principal Component %i. " % idx + "Variance Explained %2.2f" % np.multiply(eigenValues[idx], 100))
	
		img =  components[idx,:]
		img = img.reshape(original_shape[0:2])		
		
		ax1.plot(scores[:,idx])
		plt.sca(ax1)
		plt.xticks([0, 17, 34, 51, 68, 85, 102], ['0', '10', '20', '30', '40', '50', '60'])
		ax1.grid(True)
		imgplot = ax2.imshow(img, cmap=mpl.colormaps['gist_rainbow'])
		cb = plt.colorbar(imgplot)
		
		fig.canvas.draw()
		key = plt.waitforbuttonpress()
		
		if key == False:
			break
		plt.show()
		plt.close(fig)

def computePCA(data):
    
    with threadpool_limits(limits=1):
        pca = PCA()
        pca.fit(data)
        components = pca.components_
        eigenValues = pca.explained_variance_ratio_
        scores = pca.fit_transform(data)
	
        return components, eigenValues, scores

#%%
hand_lesion = np.load("/home/nipun/Documents/Uni_Malta/Alive/Alive/DataExtraction/seq30.npz")
#%%


hand_lesion = hand_lesion['skin'][:,:,:100]  # W H F


original_shape = hand_lesion.shape


hand_lesion = hand_lesion.reshape(hand_lesion.shape[0]*hand_lesion.shape[1],-1).T


components, eigenValues, scores = computePCA(hand_lesion)


plotSingle(components,eigenValues,scores,original_shape)


# %%
