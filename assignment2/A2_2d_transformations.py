import cv2
import numpy as np
import time
import keyboard
import tkinter as tk
import copy


def draw_arrow_line(img, size, center):
    img = copy.deepcopy(img)
    color = (0, 0, 0)  # 검정색
    thickness = 2

    start_x = (0, center[1])
    end_x = (size, center[1])
    cv2.arrowedLine(img, start_x, end_x, color, thickness,tipLength=0.02)

    start_y = (center[0], size)
    end_y = (center[0], 0)
    cv2.arrowedLine(img, start_y, end_y, color, thickness,tipLength=0.02)

    return img

def get_transformed_image(img, M):

    result = np.ones((801, 801), dtype=np.uint8) * 255

    center = [400, 400]

    # transformation
    x, y = img.shape
    x_offset = center[0] - x // 2
    y_offset = center[1] - y // 2

    for j in range(x):
        for i in range(y):
            # nx = M[0, 0] * i + M[0, 1] * j + M[0, 2] + x_offset + x_center
            # ny = M[1, 0] * i + M[1, 1] * j + M[1, 2] + y_offset + y_center
            nx = M[0, 0] * i + M[0, 1] * j + M[0, 2] + x_offset
            ny = M[1, 0] * i + M[1, 1] * j + M[1, 2] + y_offset
            
            if 0 < nx < 801 and 0 < ny < 801:
                nx = int(nx)
                ny = int(ny)
                result[ny, nx] = img[j][i]

    return result

def get_transformation_matrix(type):
    
    result = None

    if type == 'a':
        # Move to the left by 5 pixels
        result = np.float32([[1, 0, -5],
                             [0, 1, 0],
                             [0, 0, 1]])
        
    elif type == 'd':
        # Move to the right by 5 pixels
        result = np.float32([[1, 0, 5],
                             [0, 1, 0],
                             [0, 0, 1]])
    elif type == 'w':
        # Move to the upward by 5 pixels
        result = np.float32([[1, 0, 0],
                             [0, 1, -5],
                             [0, 0, 1]])
    elif type == 's':
        # Move to the downward by 5 pixel
        result = np.float32([[1, 0, 0],
                             [0, 1, -5],
                             [0, 0, 1]])
        
    elif type == 'r':
        # Rotate counter-clockwise by 5 degrees
        a = np.cos(np.radians(5))
        b = np.sin(np.radians(5))
        result = np.float32([[a ,b ,((1-a)-b)*400],
                             [-b, a, (b + (1 - a))*400],
                             [0,0,1]])
    
    elif type == 't':
        # Rotate clockwise by 5 degrees
        a = np.cos(np.radians(-5))
        b = np.sin(np.radians(-5))
        result = np.float32([[a ,b ,((1-a)-b)*400],
                             [-b, a, (b + (1 - a))*400],
                             [0,0,1]])
        
    elif type == 'f':
        # Flip across y axis
        result = np.float32([[-1, 0, 800],
                             [0, 1, 0],
                             [0, 0, 1]])

    elif type == 'g':
        # Flip across x axis
        result = np.float32([[1, 0, 0],
                             [0, -1, 800],
                             [0, 0, 1]])
    
    elif type == 'x':
        # Shirnk the size by 5% along to x direction 
        result = np.float32([[0.95, 0, 0.05 * 400],
                             [0, 1, 0],
                             [0, 0, 1]])
    
    elif type == 'c': 
        # Enlarge the size by 5% along to x direction
        result = np.float32([[1.05, 0, -0.05*400],
                             [0, 1, 0],
                             [0, 0, 1]])
        
    elif type == 'y':
        # Shirnk the size by 5% along to y direction 
        result = np.float32([[1, 0, 0],
                             [0, 0.95, 0.05 * 400],
                             [0, 0, 1]])
    
    elif type == 'u': 
        # Enlarge the size by 5% along to y direction
        result = np.float32([[1, 0, 0],
                             [0, 1.05,  -0.05*400],
                             [0, 0, 1]])

    return result
        
def interactive_2D_transformation(img):

    key = ""
    
    init_matrix = np.float32([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]])
    
    init_img = get_transformed_image(img, init_matrix)

    img = init_img

    while True:
        key = keyboard.read_key()

        print("You press", key)
        if key == "q":
            break

        elif key == "h":
            img = init_img

        else:
            M = get_transformation_matrix(key)
            if M is None:
                print("You Press Wrong Button! Try Again")
            img = get_transformed_image(img, M)

        arrowed_image = draw_arrow_line(img,801,[400,400])

        cv2.imshow('Transformed Image',arrowed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img

if __name__ == '__main__':
    img = cv2.imread('CV_Assignment_2_Images\smile.png',cv2.IMREAD_GRAYSCALE)
    result = interactive_2D_transformation(img)

    