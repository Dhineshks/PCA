from sklearn import datasets
from src.pca import transform

data = datasets.load_iris()

x = data.data
y = data.target

x_projected = transform(x)

print(x.shape)
print(x_projected.shape)
