import numpy as np
import cv2
import compute_avg_reproj_error 
import time

def compute_F_raw(M):

    n, _ = M.shape
    # 1. M*9 Matrix 만들기
    A = np.zeros((M.shape[0],9))

    for i in range(n):
        x1, y1, x2, y2 = M[i][0], M[i][1],M[i][2],M[i][3],
        A[i] = [x1*x2, x2*y1, x2, x1*y2, y1*y2, y2, x1,y1, 1]

    _,_,V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)

    return F

def compute_F_norm(M):

    mean = np.mean(M, axis=0)

    trans_M = M-mean

    x1, y1, x2, y2 = trans_M[:,0], trans_M[:,1],trans_M[:,2],trans_M[:,3]

    # print(np.max(np.sqrt(x1**2 + y1 **2)))
    # scaling을 위해서.. 
    scaling_1 = np.sqrt(2)/np.max(np.sqrt(x1**2 + y1 **2))
    scaling_2 = np.sqrt(2)/np.max(np.sqrt(x2**2 + y2 **2))

    # print(np.max(trans_M,axis=0))

    # print(trans_M[:0] * scaling_1)
    trans_M[:,0] = trans_M[:,0] * scaling_1
    trans_M[:,1] = trans_M[:,1] * scaling_1
    trans_M[:,2] = trans_M[:,2] * scaling_2
    trans_M[:,3] = trans_M[:,3] * scaling_2

    # print(np.max(trans_M,axis=0))

    trans_F = compute_F_raw(trans_M)

    scaling_M_1 = np.array([[scaling_1, 0 , -mean[0]*scaling_1],
                           [0,scaling_1, -mean[1]*scaling_1],
                           [0,0,1]])
    
    scaling_M_2 = np.array([[scaling_2, 0 , -mean[2]*scaling_2],
                           [0, scaling_2, -mean[3]*scaling_2],
                           [0,0,1]])
    
    F = np.dot(np.dot(scaling_M_2.T, trans_F), scaling_M_1)
    
    return F

def comput_F_mine(M):

    # ransac algorithm
    # inlier, outlier를 어떻게 계산하지... > Compute avg reporj error를 best로 만들면 되지 않을까

    start = time.time()

    best_F = compute_F_norm(M)
    min_score = compute_avg_reproj_error.compute_avg_reproj_error(M, best_F)
    while(True):
        end = time.time()

        if end-start > 4.5:
            break

        idx = np.random.choice(len(M), 50)

        # print(idx)
        rand_M = M[idx]

        rand_F = compute_F_norm(rand_M)

        score = compute_avg_reproj_error.compute_avg_reproj_error(M, rand_F)

        if score < min_score:
            min_score = score
            best_F = rand_F
    
    return best_F
     

def compute_F_three(img1, img2, M):
    raw_F = compute_F_raw(M)
    norm_F = compute_F_norm(M)
    mine_F = comput_F_mine(M)

    raw_score = compute_avg_reproj_error.compute_avg_reproj_error(M, raw_F)
    norm_score = compute_avg_reproj_error.compute_avg_reproj_error(M, norm_F)
    mine_score = compute_avg_reproj_error.compute_avg_reproj_error(M, mine_F)

    print(f"    Raw = {raw_score}")
    print(f"    Norm = {norm_score}")
    print(f"    Mine = {mine_score}\n")

    return mine_F

def draw_epipolar_line(img1, img2, F, point1, point2):

    r, c = img1.shape

    point1 = np.array([[point1[0,0], point1[0,1], 1],
                      [point1[1,0], point1[1,1], 1],
                      [point1[2,0], point1[2,1], 1]])
    
    point2 = np.array([[point2[0,0], point2[0,1], 1],
                      [point2[1,0], point2[1,1], 1],
                      [point2[2,0], point2[2,1], 1]])
    
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) 
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) 

    point2_to_img1 = np.dot(F.T,point2.T).T
    point1_to_img2 = np.dot(F, point1.T).T

    colors = [(255,0,0), (0,255,0), (0,0,255)]
    
    img1_copy = img1.copy()
    img2_copy = img2.copy()

    for map_point, p1, p2, color in zip(point2_to_img1, point1, point2, colors):
        
        # orut
        x1 = 0
        y1 = -map_point[2]/map_point[1]
        x2 = c
        y2 = -(map_point[2] + map_point[0] * c) / map_point[1]

        img1_copy = cv2.line(img1_copy, (x1,int(y1)),(x2,int(y2)),color,1)
        img1_copy = cv2.circle(img1_copy, (int(p1[0]), int(p1[1])), 3, color, 2)
        img2_copy = cv2.circle(img2_copy, (int(p2[0]), int(p2[1])), 3, color, 2)

    for map_point, color in zip(point1_to_img2, colors):
        x1 = 0
        y1 = -map_point[2]/map_point[1]
        x2 = c
        y2 = - (map_point[2] + map_point[0] * c) / map_point[1]
        # print(x1,y1,x2,y2)
        img2_copy = cv2.line(img2_copy, (x1,int(y1)),(x2,int(y2)),color,1)

    img = np.concatenate((img1_copy, img2_copy), axis=1)
    cv2.imshow('result', img)
    return img
    
   
def visualize_epipolar(img1, img2, M, F):
    idx = np.random.choice(len(M), 3)

    rand_M = M[idx]

    # print(rand_M)
    # print(rand_M[:,:2],rand_M[:,2:])
    # result = np.concatenate((img1, img2), axis=1)

    # cv2.imshow('result', result)
    key = 0
    while True:
        # print("You Press", key)
        if key == 'q':
            break
        
        else:
            idx = np.random.choice(len(M), 3)
            rand_M = M[idx]
            draw_epipolar_line(img1, img2, F, rand_M[:,:2],rand_M[:,2:])
        
        key = chr(cv2.waitKey())

if __name__ == "__main__":

    # print("----- For Temple Image -----")
    print("Average Reprojection Errors (temple1.png and temple2.png)")
    temple1 = cv2.imread('CV_Assignment_3_Data/temple1.png',cv2.IMREAD_GRAYSCALE)
    temple2 = cv2.imread('CV_Assignment_3_Data/temple2.png',cv2.IMREAD_GRAYSCALE)
    temple_M = np.loadtxt("CV_Assignment_3_Data/temple_matches.txt")
    temple_F = compute_F_three(temple1, temple2, temple_M)

    print("----- For house Image -----")
    print("Average Reprojection Errors (house1.jpg and house2.jpg)")
    house1 = cv2.imread('CV_Assignment_3_Data/house1.jpg',cv2.IMREAD_GRAYSCALE)
    house2 = cv2.imread('CV_Assignment_3_Data/house2.jpg',cv2.IMREAD_GRAYSCALE)
    house_M = np.loadtxt("CV_Assignment_3_Data/house_matches.txt")
    house_F = compute_F_three(house1, house2, house_M)

    # print("----- For Library Image -----")
    print("Average Reprojection Errors (library1.jpg and library2.jpg)")
    library1 = cv2.imread('CV_Assignment_3_Data/library1.jpg',cv2.IMREAD_GRAYSCALE)
    library2 = cv2.imread('CV_Assignment_3_Data/library2.jpg',cv2.IMREAD_GRAYSCALE)
    library_M = np.loadtxt("CV_Assignment_3_Data/library_matches.txt")
    library_F = compute_F_three(library1, library2, library_M)

    visualize_epipolar(temple1, temple2, temple_M,temple_F)
    visualize_epipolar(house1, house2, house_M,house_F)
    visualize_epipolar(library1, library2, library_M,library_F)