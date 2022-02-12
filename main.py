import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.spatial import distance
import time

data = []
density_map = []

SETS = 4

POINTS = 100

MEAN = [[0,0],
        [4,0],
        [0,4],
        [5,4]]

COV =  [[[1  ,0  ], 
         [0  ,1  ]], 
        [[1  ,0.2], 
         [0.2,1.5]], 
        [[1  ,0.4], 
         [0.4,1.1]], 
        [[0.3,0.2], 
         [0.2,0.5]]]

# Range of viewport
XO, YO = -3, 8

def euclidean_distance(i,j):
  return ((j[0] - i[0])**2 + (j[1] - i[1])**2)**(1/2)


# def min_max_distance(dataset):
#   start_time = time.time()
#   max_d = float("-inf")
#   min_d = float("inf")
#   max_between = []
#   min_between = []
#   for i in dataset:
#     for j in dataset:
#       if (i[0] == j[0] and i[1] == j[1]): continue
#       dist = euclidean_distance(i, j)

#       if dist > max_d:
#         max_d = dist
#         max_between = [i,j]

#       if dist < min_d:
#         min_d = dist
#         min_between = [i,j]

#   max_between = np.array(max_between)
#   min_between = np.array(min_between)
#   print("\n--- %s seconds for computation ---" % (time.time() - start_time))
#   return max_d, max_between, min_d, min_between


def min_max_euclid(dataset):
  start_time = time.time()
  distances = distance.cdist(dataset,dataset,metric='euclidean')
  minvals = []
  val = []
  for row in distances:
    val.append(np.max(row))
    minvals.append(np.partition(row,1)[1])

  valuemax = np.amax(distances)
  valuemin = np.min(np.array(minvals))

  print("\n--- %s seconds for computation --- " % (time.time() - start_time))
  return (valuemax, np.where(distances == valuemax)[0], valuemin, np.where(distances == valuemin)[0])


# 2D Scatter and Contour
plt.style.use("seaborn")
# plt.style.use("ggplot")
xx, yy = np.meshgrid(*np.linspace(XO,YO,POINTS)[np.newaxis,...].repeat(2,0))
# gridspace.shape is (100,100)
gridspace = np.array([xx,yy]).transpose((1,2,0)).reshape(-1,2)

fig2d, ax2D = plt.subplots(figsize=(14,7))
ax2D0 = plt.subplot2grid(shape=(2,4), loc=(0,0), rowspan=2, colspan=2)
ax2D1 = plt.subplot2grid(shape=(2,4), loc=(0,2))
ax2D2 = plt.subplot2grid(shape=(2,4), loc=(0,3))
ax2D3 = plt.subplot2grid(shape=(2,4), loc=(1,2))
ax2D4 = plt.subplot2grid(shape=(2,4), loc=(1,3))

for i in range(SETS):
  np.random.seed(42)
  vectors = np.random.multivariate_normal(mean=MEAN[i], cov=COV[i], size=POINTS)
  data.append([vectors])
  z = multivariate_normal.pdf(gridspace, mean=MEAN[i], cov=COV[i])
  # print("z:\n\n", z)
  z = z.reshape(POINTS,POINTS)
  density_map.append(z)

  cs = ax2D0.contour(xx,yy,z)
  plt.clabel(cs)
  ax2D0.scatter(*vectors.T, s=8)

# Indivisual Set Contours
plt.style.use("default")
density_map = np.array(density_map)
data = np.array(data)
ax2D1.contourf(xx, yy, density_map[0])
ax2D2.contourf(xx, yy, density_map[1])
ax2D3.contourf(xx, yy, density_map[2])
ax2D4.contourf(xx, yy, density_map[3])

# Minimum and Maximum Plot
maximum, idmax, minimum, idmin = min_max_euclid(data.reshape(-1,2))
max_between = np.array([data.reshape(-1,2)[idmax[0]], data.reshape(-1,2)[idmax[1]]])
min_between = np.array([data.reshape(-1,2)[idmin[1]], data.reshape(-1,2)[idmin[1]]])
print(f"\nMaximum: {maximum} \n>> Between \n{data.reshape(-1,2)[idmax[0]]} and {data.reshape(-1,2)[idmax[1]]}\n")
print(f"Minimum: {minimum} \n>> Between \n{data.reshape(-1,2)[idmin[0]]} and {data.reshape(-1,2)[idmin[1]]}\n")

# maximum, max_between, minimum, min_between = min_max_distance(data.reshape(-1,2))
# print(f"\nMaximum: {maximum} \n>> Between \n{max_between}\n")
# print(f"Minimum: {minimum} \n>> Between \n{min_between}\n")

x_max = [max_between[0,0], max_between[1,0]]
y_min = [max_between[0,1], max_between[1,1]]
ax2D0.plot(x_max, y_min, linestyle="dashed", label="Maximum: {:.3f}".format(maximum))
x_min = [min_between[0,0], min_between[1,0]]
y_min = [min_between[0,1], min_between[1,1]]
ax2D0.plot(x_min, y_min, lw=5, c="black", label="Minimum: {:.3f}".format(minimum))
ax2D0.legend()

# 3D Scatter
plt.style.use("default")
fig3d = plt.figure(figsize=(6,6))
ax3D = fig3d.add_subplot(111, projection='3d')
for i in range(SETS):
  x_,y_ = data[i].T
  z_ = multivariate_normal.pdf(data[i], mean=MEAN[i], cov=COV[i])
  ax3D.scatter(x_, y_, z_, marker='o', s=5)

ax3D.set_xlabel('X')
ax3D.set_ylabel('Y')
ax3D.set_zlabel('Gaussian Probablity Density')

plt.show()