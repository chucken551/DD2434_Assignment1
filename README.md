# DD2434_Assignment1
4 libraries are required:
- sklearn
- numpy
- scipy
- matplotlib

Make sure you set the variable "DIR" to the directory of the zoo dataset. If the file is in the same directory, "zoo.data" will work.
Functions and classes:
- DataGenerator(), A class that gather the zoo data and remove unwanted parts. Saves the data as attributes.
- get_data(), method in the DataGenerator() class that returns the data stored in it's attributes.
- get_colorType(), method in the DataGenerator() class that return the corresponding type of animal to the data.
- PCA_Generator(), function that takes input data and return a 2 dimensional projection.
- mds_Generator(), function that takes input data and return a 2 dimensional projection.
- isomap_generator(), function that takes input data and number of nearest neighbors and return a 2 dimensional projection.
- distance_Matrix(), function that take in a matrix and return the matrix with the distance to each corresponding point.
- plot_graphs(), function that take in data and title and plots scatter graphs.
- main(), function that has no input and create objects and call functions.


Run the code to plot three graphs, PCA, MDS and ISOMAP. For ISOMAP the number of K nearest neighbors can be change in the
top of main(), line 157.
