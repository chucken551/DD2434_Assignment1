import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#from sklearn.manifold import MDS
#from scipy.spatial import distance_matrix
from numpy import linalg
#from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import floyd_warshall
#from sklearn.manifold import Isomap

class DataGenerator:
    def __init__(self, filename):
        self.filename = filename
        self.initaldata = None
        self.colorType = None
        self.animals = None
        self.data_retriver()
    def data_retriver(self):
        data_list = []
        with open(self.filename, "r") as datafile:
            reader = datafile
            color_list =[]
            data_matrix=[]
            animal_list=[]
            for row in reader:
                row=row.replace("\n", "")
                row=row.split(",")
                temp_list=[]
                animal_list.append(row[0])
                for i in range(len(row)):

                    if 0<i<17:
                        temp_list.append(int(row[i]))

                    if i == 17:

                        color_list.append(int(row[i]))

                data_matrix.append(temp_list)
        self.initaldata = data_matrix
        self.colorType = color_list
        self.animals = animal_list
    def get_data(self):
        return self.initaldata

    def get_colorType(self):
        return self.colorType

    def get_animal(self):
        return self.animals
def PCA_Generator(dataset):
    #Transforming data so that rows are dimensions and colums are data points
    Y = np.transpose(dataset)
    n, m = dataset.shape
    #Centerer data
    Y = Y - (1/n) * np.dot(Y, np.ones((n,n)))
    Y = Y.astype('float')
    U, S, V = np.linalg.svd(Y)
    index = np.argsort(S)
    index = np.flip(index)
    #Index of the largest singular values, sort U
    U = U[:,index]
    U_k = U[:,(0,1)]
    U_k = np.transpose(U_k)
    #Obtain the projection point matrix
    X = np.transpose(U_k.dot(Y))
    return X
"""
def PCA(dataset):
    pca = PCA(n_components=2, svd_solver='auto')
    #Use the built in function to provide the PCA dimensionality reduction with SVD solver for the dataset
    X_r = pca.fit(dataset).transform(dataset)
    return X_r
"""

def mds_Generator(dataset):
    #Fetch the distance matrix D from the dataset.
    D = distance_Matrix(dataset)
    #Calculating the center matrix J
    J = np.identity(D.shape[0]) - (1/(D.shape[0]))*(np.ones((D.shape[0], D.shape[0])))
    #Applying double centering B
    B = -(1/2)*J.dot(D**2).dot(J)
    #Fetching eigenvalues and eigenvectors from the doublecentered matrix.
    eig_val, eig_vec = np.linalg.eigh(B)
    "argsort() return the indices of the array that would sort an array."
    #Sorting and flipping so that largest eigenvalues are first.
    index = np.argsort(eig_val)
    index = np.flip(index)
    eig_vec = eig_vec[:,index]
    eig_val = eig_val[index]
    #Use the m largest eigenvalues, m is the desired dimension. In our case m is the first 2 values.
    E_m = eig_vec[:, [0,1]]
    #Create diagonal matrix of the corresponding eigenvectors for the m eigenvalues. In our case a 2x101 matrix.
    Lambda_m = np.diag(eig_val[[0,1]])
    #Create the decomposed matrix with the disired dimensionality reduction. In our case a 101x2 matrix.
    X = np.dot(E_m, np.sqrt(Lambda_m))
    return X
"""
def isomap(dataset):
    X = dataset
    embedding = Isomap(n_neighbors=50, n_components = 2)
    X_tranformed = embedding.fit_transform(X)
    return X_tranformed
"""
def isomap_generator(K, dataset):
    #K is the number of nearest nieghbours
    X = dataset
    #Create the Distance matrix of the dataset.
    D = distance_Matrix(dataset)
    index = D.argsort()
    #Create neighbours that are the index for the sorted K euclidan closest neighbours.
    neighbours = index[:, :K+1]
    #Create a new distance matrix with only the K euclidian closest neighbours, the rest are set to infinity.
    H = np.ones((X.shape[0], X.shape[0]), dtype='float')*np.inf
    for i in range(X.shape[0]):
        H[i, neighbours[i, :]] = D[i, neighbours[i, :]]
    #Use the floyd Warshall algorithm to find the closest neighbors.
    D_graph = floyd_warshall(H)**2
    #Same as for the MDS
    J = np.identity(D_graph.shape[0]) - (1/(D_graph.shape[0]))*(np.ones((D_graph.shape[0], D_graph.shape[0])))
    B = -(1/2)*J.dot(D_graph**2).dot(J)
    eig_val, eig_vec = np.linalg.eigh(B)
    index = np.argsort(eig_val)
    index = np.flip(index)
    eig_vec = eig_vec[:,index]
    eig_val = eig_val[index]
    E_m = eig_vec[:, [0,1]]
    Lambda_m = np.diag(eig_val[[0,1]])
    X = np.dot(E_m, np.sqrt(Lambda_m))
    return X

def distance_Matrix(matrix):
    #Creating empty matrix
    distance_matrix = np.zeros((matrix.shape[0], matrix.shape[0]))
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            #Calculating the euclidian distance to for every sample to another and placing them in the matrix.
            distance_matrix[i,j]= np.sqrt(np.sum((matrix[i]-matrix[j])**2))
    return distance_matrix

def plot_graphs(data, matrix, type):
    X= matrix
    plt.figure()
    colors = {1:'navy', 2:'turquoise', 3:'darkorange', 4:'green', 5:'blue', 6:'red', 7:'yellow'}
    recs = []
    p = []
    for i in range(len(data.get_animal())):
        scatter = plt.scatter(X[i,0], X[i,1], color=colors[data.get_colorType()[i]], alpha=0.8, lw=2)
        if colors[data.get_colorType()[i]] not in recs:
            recs.append(colors[data.get_colorType()[i]])
            p.append(scatter)
    plt.legend(p,('1','2','3','4','5','6','7'), scatterpoints=1, title='Type')
    plt.title(type +' of Animals')


def main():
    K = 25
    DIR = "zoo.data"
    data = DataGenerator(DIR)
    dataset = np.array(data.get_data())
    data_color = data.get_colorType()
    PCA = PCA_Generator(dataset)
    MDS = mds_Generator(dataset)
    Isomap = isomap_generator(K, dataset)
    plot_graphs(data, PCA, "PCA")
    plot_graphs(data, MDS, "MDS")
    plot_graphs(data, Isomap, 'Isomap K='+ str(K))
    plt.show()
main()
