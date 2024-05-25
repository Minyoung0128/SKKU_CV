import cv2
import numpy as np
import time
import keyboard
import tkinter as tk
import copy

def get_transformed_image(img, M):

    result = np.ones((801, 801), dtype=np.uint8) * 255

    center_x = 400
    center_y = 400

    # center 변하는거도 맞춰줘야 함

    dim3_vect = [0,0,1]
    center_x, center_y, k = np.dot(M, dim3_vect)

    center_x = round(center_x) + 400
    center_y = round(center_y) + 400

    # transformation
    x, y = img.shape

    x_half = x // 2
    y_half = y // 2

    x_offset = center_x - x_half
    y_offset = center_y - y_half
    
    for i in range(x_offset, center_x + x_half + 1):
        for j in range(y_offset, center_y + y_half +1):
            dim3_v = np.array([i - center_x, j - center_y, 1])
            nx, ny, _ = np.dot(M, dim3_v)

            nx = int(nx)
            ny = int(ny)

            nx += center_x
            ny += center_y

            # if nx < 0 or nx > 801 or ny < 0 or ny > 801 :
            #     print("Out Of Index !!")
            #     return None

            if img[i - x_offset][j - y_offset] < 255:
                    result[nx][ny] = img[i - x_offset][j - y_offset]


    cv2.arrowedLine(result, (0, 400),  (801, 400), (0, 0, 0), 2,tipLength=0.02)
    cv2.arrowedLine(result, (400, 801), (400, 0), (0, 0, 0), 2,tipLength=0.02)

    return result

       
def interactive_2D_transformation(img):

    key_a = np.float32([[1, 0, 0],
                        [0, 1, -5],
                        [0, 0, 1]])
    
    key_d = np.float32([[1, 0, 0],
                        [0, 1, 5],
                        [0, 0, 1]])
    
    key_w = np.float32([[1, 0, -5],
                        [0, 1, 0],
                        [0, 0, 1]])
    
    key_s = np.float32([[1, 0, 5],
                        [0, 1, 0],
                        [0, 0, 1]])
    
    key_r = np.float32([[np.cos(np.radians(5)), -np.sin(np.radians(5)) ,0],
                        [np.sin(np.radians(5)), np.cos(np.radians(5)), 0],
                        [0,0,1]])

    
    key_t = np.float32([[np.cos(np.radians(5)) ,np.sin(np.radians(5)) ,0],
                        [-np.sin(np.radians(5)), np.cos(np.radians(5)), 0],
                        [0,0,1]])
    
    key_f = np.float32([[1, 0, 0],
                        [0, -1, 0],
                        [0, 0, 1]])
    
    key_g = np.float32([[-1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
    
    key_x = np.float32([[1, 0, 0],
                        [0, 0.95, 0],
                        [0, 0, 1]])

    key_c = np.float32([[1, 0, 0],
                        [0, 1.05,0],
                        [0, 0, 1]])
    

    key_y = np.float32([[0.95, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])


    key_u = np.float32([[1.05, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
    

    M = np.float32([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    
    result = get_transformed_image(img, M)

    cv2.imshow("Transformed Image",result)

    while True:
        key = chr(cv2.waitKey())
        # print("You Press", key)
        if key == 'a':
        # Move to the left by 5 pixels
            M = np.dot(key_a, M) 
        
        elif key == 'd':
            # Move to the right by 5 pixels
            M = np.dot(key_d, M)
        elif key == 'w':
            # Move to the upward by 5 pixels
            M = np.dot(key_w, M)
        elif key == 's':
            # Move to the downward by 5 pixel
            M = np.dot(key_s, M)
            
        elif key == 'r':
            # Rotate counter-clockwise by 5 degrees
            M = np.dot(key_r, M)
        elif key == 't':
            # Rotate clockwise by 5 degrees
            M = np.dot(key_t, M)
            
        elif key == 'f':
           M = np.dot(key_f, M)
            
        elif key == 'g':
            M = np.dot(key_g, M)
            
        elif key == 'x':
            # Shirnk the size by 5% along to x direction 
            M = np.dot(key_x, M)
        
        elif key == 'c': 
            # Enlarge the size by 5% along to x direction
            M = np.dot(key_c, M)
            
        elif key == 'y':
            # Shirnk the size by 5% along to y direction
            M = np.dot(key_y, M)
        
        elif key == 'u': 
            # Enlarge the size by 5% along to y direction
            M = np.dot(key_u, M)

        elif key == "q":
            break

        elif key == "h":
            M = np.float32([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
        
        else:
            print("You Press Wrong Button! Try Again")
            continue

        result = get_transformed_image(img, M)

        cv2.imshow("Transformed Image",result)

    return img

if __name__ == '__main__':
    img = cv2.imread('CV_Assignment_2_Images\smile.png',cv2.IMREAD_GRAYSCALE)
    result = interactive_2D_transformation(img)

