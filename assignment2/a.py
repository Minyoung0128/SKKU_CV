import numpy as np

m = [[5,3,0,1],[4,0,0,1],[1,1,0,5],[1,0,0,4],[0,1,5,4]]
u,s,v = np.linalg.svd(m)
print(u)
print(s)
print(v)

print("Matrix ")
print(u[:, :2] @ np.diag(s[:2]) @ v[:2, :] )