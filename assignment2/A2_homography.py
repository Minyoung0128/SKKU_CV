import cv2
import numpy as np
from numpy import sqrt
import time
import keyboard
import copy
import math
import random
def find_x_y_cor(src1, src2):
    size_n = 35

    m_a = np.zeros((size_n, 2))
    m_b = np.zeros((size_n, 2))
    orb = cv2.ORB_create()

    kp1 = orb.detect(src1, None)
    kp1, des1 = orb.compute(src1, kp1)

    kp2 = orb.detect(src2, None)
    kp2, des2 = orb.compute(src2, kp2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    dic = {}
    dic_list = []

    # list 자료형 matches
    # matches 인스턴스의 attribute 인 distance 길이를 기준으로 정렬
    matches = sorted(matches, key=lambda x: x.distance)

    for i in range(size_n):
        dic = {'src': matches[i].queryIdx, 'des': matches[i].trainIdx}
        dic_list.append(dic)
    i = 0
    for dic in dic_list:
        m_a[i][0] = kp1[dic['src']].pt[0]
        m_a[i][1] = kp1[dic['src']].pt[1]
        m_b[i][0] = kp2[dic['des']].pt[0]
        m_b[i][1] = kp2[dic['des']].pt[1]

        # print(m_b[i][0])
        i += 1
    return m_a, m_b

def get_orb(img):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    return orb.detectAndCompute(img, None)

def hammingDistance(a, b):
    r = (1 << np.arange(8))[:,None]
    return np.count_nonzero((np.bitwise_xor(a,b) & r) != 0)

def get_match_pair(matches, keypoints1, keypoints2):
    srcP = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    destP = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return srcP, destP 

def find_best_match(des1, des2):
    match = []
    for i in range(des1.shape[0]):
        best_idx = -1
        best_dist = np.Inf

        for j in range(des2.shape[0]):
            d1 = des1[i]
            d2 = des2[j]

            # dist = hammingDistance(d1,d2)
            dist = cv2.norm(d1,d2,cv2.NORM_HAMMING)

            if best_dist > dist:
                best_dist = dist
                best_idx = j

        # 여기까지 오면 i번째 match 와 제일 좋은 match를 가져옴
        if best_dist < 50:
            # print("Here")
            match.append(cv2.DMatch(i, best_idx, best_dist))

    # 모든 match들 중에 상위 10개만 뽑기 
    match = sorted(match, key = lambda x:x.distance)

    return match

def get_match_pair(matches, keypoints1, keypoints2):
    # srcP = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    # destP = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    srcP = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    destP = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    return srcP, destP 

def show_match_pair(img1, img2):
    kp1, des1 = get_orb(img1)
    kp2, des2 = get_orb(img2)

    matches = find_best_match(des1, des2)

    dst = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None,flags=2)

    srcP, destP = get_match_pair(matches[:40], kp1, kp2)

    # 출력
    cv2.imshow("ORB",dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return matches, srcP, destP

def normalize(array):
    # 열별로 normalization

    N = len(array)
    # 입력 배열을 2차원 배열로 변환
    array_reshaped = array.reshape(-1, 2)

    # 열별 평균 계산
    mean = np.mean(array_reshaped, axis=0)
    array_centered = array_reshaped - mean

    # Scaling
    scaling_factor = np.sqrt(2) / np.max(np.linalg.norm(array_centered, axis=1))

    array_normalized = array_centered * scaling_factor

    M = np.float32([[scaling_factor, 0, -mean[0]*scaling_factor],
                    [0, scaling_factor, -mean[1]*scaling_factor],
                    [0, 0, 1]])
    
    return array_normalized, M

def compute_normalize_homography(norm_src, norm_dest):

    A = np.zeros((2 * norm_src.shape[0],9))
    
    for i in range(norm_src.shape[0]):
        
        x_src = norm_src[i, 0]
        y_src = norm_src[i, 1]
        x_dest = norm_dest[i, 0]
        y_dest = norm_dest[i, 1]

        A[i*2] = [-x_src, -y_src,-1, 0, 0, 0, x_src*x_dest, y_src*x_dest, x_dest]
        A[i*2 + 1] = [0, 0, 0,-x_src, -y_src,-1,x_src*y_dest, y_src*y_dest, y_dest]
    
    # print(A)
    _,_,V = np.linalg.svd(A)
    H_norm = V[-1].reshape(3,3)
    return H_norm/H_norm[-1][-1]

def compute_homography(srcP, destP):

    # print(cv2.findHomography(srcP, destP, cv2.LMEDS))
    answ_h,_= cv2.findHomography(srcP, destP, 0)
    # print("Answer")
    # print(answ_h)
    # Normalize feature point
    norm_srcP, T_src = normalize(srcP)
    norm_destP, T_dest = normalize(destP)

    H_norm = compute_normalize_homography(norm_srcP, norm_destP)
    
    # print(H_norm.shape)
    H = np.dot(np.dot(np.linalg.inv(T_dest),H_norm),T_src)
    # print("Before normalize")
    # print(H)
    # print(H /H[-1,-1])
    return H /H[-1,-1]
    # print(H)
    # return H
    # return answ_h

def get_dist(p1, p2, H):
    p1 = np.array([[p1[0], p1[1], 1]])
    p2 = np.array([[p2[0], p2[1], 1]])

    des_p1 = np.dot(H, p1.T)
    des_p1 = des_p1 / des_p1[-1]

    error = des_p1.T - p2

    return np.linalg.norm(error)

def wrap_image(source_img, destination_img, H):
    # cv2.warpPerspective를 사용하여 래핑
    h, w = destination_img.shape
    warped_img = cv2.warpPerspective(source_img, H, (w, h))

    # 래핑된 이미지를 cv_desk.png와 합성
    composed_img = cv2.addWeighted(destination_img, 0.5, warped_img, 0.5, 0)

    return warped_img, composed_img

def compute_homography_ransac(srcP, destP, th):
    
    max_inliers = []
    finalH = None

    start = time.time()
    while(True):
        end = time.time()

        if end-start > 3:
            break
        idx = np.random.choice(len(srcP), 4)

        rand_src = srcP[idx]        
        rand_dest = destP[idx]

        H = compute_homography(rand_src, rand_dest)

        if H is None:
            continue

        inliers = []

        for i in range(srcP.shape[0]):
            dist = get_dist(srcP[i], destP[i], H)

            if dist < th:
                inliers.append([srcP[i], destP[i]])

        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            finalH = H

        if len(max_inliers) > (10 * th):
            break

    return finalH            

def image_stitch(img1, img2):
    kp1, des1 = get_orb(img1)
    kp2, des2 = get_orb(img2)

    matches = find_best_match(des1, des2)
    srcP, destP = get_match_pair(matches[:15], kp1, kp2)

    H = compute_homography_ransac(srcP, destP, 1)

    result = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    # cv2.imshow('warped', result)
    result[0: img1.shape[0], 0: img2.shape[1]] = img2

    # image blending
    # dst = result[0:imageB.shape[0], imageB.shape[1] - 5: imageB.shape[1] + 5]
    for i in range(10):
        tmp1 = result[0:img2.shape[0], img2.shape[1] - i]
        tmp2 = result[0:img2.shape[0], img2.shape[1] + i]
        result[0:img2.shape[0], img2.shape[1] - i: img2.shape[1] - i + 1 ] = cv2.addWeighted(tmp1, 0.7, tmp2, 0.3, 0)
        result[0:img2.shape[0], img2.shape[1] + i: img2.shape[1] + i + 1] = cv2.addWeighted(tmp1, 0.3, tmp2, 0.7, 0)


    cv2.imshow('blur_result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    imageA = cv2.imread('CV_Assignment_2_Images\cv_cover.jpg',cv2.IMREAD_GRAYSCALE)
    imageB = cv2.imread('CV_Assignment_2_Images\cv_desk.png',cv2.IMREAD_GRAYSCALE)
    
    match, srcP, destP = show_match_pair(imageA, imageB)

    H = compute_homography(srcP, destP)
    H_RANSAC = compute_homography_ransac(srcP,destP, 5)

    print(H_RANSAC)

    result1 = cv2.warpPerspective(imageA, H, (imageB.shape[1], imageB.shape[0]))
    result2 = cv2.warpPerspective(imageA, H, (imageB.shape[1], imageB.shape[0]))
    for i in range(imageB.shape[0]):
        for j in range(imageB.shape[1]):
            if result2[i, j] < 10:
                result2[i, j] = imageB[i, j]

    result3 = cv2.warpPerspective(imageA, H_RANSAC, (imageB.shape[1], imageB.shape[0]))
    result4 = cv2.warpPerspective(imageA, H_RANSAC, (imageB.shape[1], imageB.shape[0]))
    
    for i in range(imageB.shape[0]):
        for j in range(imageB.shape[1]):
            if result4[i, j] < 10:
                result4[i, j] = imageB[i, j]
    
    hp_cover = cv2.imread('CV_Assignment_2_Images\hp_cover.jpg',cv2.IMREAD_GRAYSCALE)
    hp_cover = cv2.resize(hp_cover, (imageA.shape[1], imageA.shape[0]))
    result5 = cv2.warpPerspective(hp_cover, H_RANSAC, (imageB.shape[1], imageB.shape[0]))
    result6 = cv2.warpPerspective(hp_cover, H_RANSAC, (imageB.shape[1], imageB.shape[0]))
    
    for i in range(imageB.shape[0]):
        for j in range(imageB.shape[1]):
            if result6[i, j] < 10:
                result6[i, j] = imageB[i, j]

    cv2.imshow('result1', result1)
    cv2.imshow('result2', result2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('result3', result3)
    cv2.imshow('result4', result4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('result5', result5)
    cv2.imshow('result6', result6)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    image10 = cv2.imread('CV_Assignment_2_Images\diamondhead-10.png',cv2.IMREAD_GRAYSCALE)
    image11 = cv2.imread('CV_Assignment_2_Images\diamondhead-11.png',cv2.IMREAD_GRAYSCALE)

    image_stitch(image11, image10)



