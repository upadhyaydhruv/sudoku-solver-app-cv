import numpy as np
import cv2 as cv
import sys
import imutils
import math

#The code below implements the sudoku solver, assuming it's a nine by nine grid (2 dimensional array - list of lists)

#Sample sudoku grid
grid = [
    [5,3,0,0,7,0,0,0,0],
    [6,0,0,1,9,5,0,0,0],
    [0,9,8,0,0,0,0,6,0],
    [8,0,0,0,6,0,0,0,3],
    [4,0,0,8,0,3,0,0,1],
    [7,0,0,0,2,0,0,0,6],
    [0,6,0,0,0,0,2,8,0],
    [0,0,0,4,1,9,0,0,5],
    [0,0,0,0,8,0,0,7,9]
    ]

# this returns whether or not a number can be placed at a given position in the grid
def possible_placed(y, x, n):
    global grid

    for i in range(0,9):
        if grid[y][i] == n:
            return False
    
    for i in range(0,9):
        if grid[i][x] == n:
            return False
    
    x0 = (x//3)*3
    y0 = (y//3)*3

    for i in range(3):
        for j in range(3):
            if grid[y0+i][x0+j]==n:
                return False
    
    return True


#This function solves the sudoku grid using a recursive bactracking algorithm

def solve():
    global grid

    for y in range(9):
        for x in range(9):
            if grid[y][x] == 0:
                for n in range(1,10):
                    if possible_placed(y,x,n):
                        grid[y][x] = n
                        solve() 
                        grid[y][x] = 0
                return
    print(np.matrix(grid))

#solve()  


def order_points(pts):
    # Step 1: Find centre of object
    center = np.mean(pts)

    # Step 2: Move coordinate system to centre of object
    shifted = pts - center

    # Step #3: Find angles subtended from centroid to each corner point
    theta = np.arctan2(shifted[:, 0], shifted[:, 1])

    # Step #4: Return vertices ordered by theta
    ind = np.argsort(theta)
    return pts[ind]


file_path = "D:\Dhruv\sudoku-solver-test2.jpg"
img = cv.imread(file_path, 0) 
cv.imshow('display', img)
k=cv.waitKey(0)
#ratio = img.shape[0] / 300.0
orig = img.copy()
img = imutils.resize(img, height=300)
if img is None:
    sys.exit("Could not read the image.")

gauss_blurred = cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT) #Applying Gaussian blur on image to get rid of unnecessary noise 

adaptive_threshold = cv.adaptiveThreshold(gauss_blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 3) #

adaptive_threshold = cv.bitwise_not(adaptive_threshold)

kernel = np.ones((2,2),np.uint8)
dilated = cv.dilate(adaptive_threshold, kernel, iterations = 1)
eroded = cv.erode(adaptive_threshold, kernel)

cnts = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c=max(cnts, key=cv.contourArea)
#cv.drawContours(img, [c], 0, (0, 255, 0), 3)


#Finding extreme four points on a contour
maxSum = 0
minSum = 1000
maxDiff = -1000
minDiff = 1000
tr = np.array([0, 0])
tl = np.array([0, 0])
br = np.array([0, 0])
bl = np.array([0, 0])

for i in range(116):
    if np.sum(c[i])>maxSum:
        maxSum = np.sum(c[i])
        br = c[i][0]
    if np.sum(c[i])<minSum:
        minSum = np.sum(c[i])
        tl = c[i][0]
    if abs(np.diff(c[i]))>maxDiff:
        maxDiff = np.diff(c[i])
        bl = c[i][0]
    if abs(np.diff(c[i]))<maxDiff:
        minDiff = np.diff(c[i])
        tr = c[i][0]

tl = (tl[0], tl[1])
tr = (tr[0], tr[1])
bl = (bl[0], bl[1])
br = (br[0], br[1])

#cv.circle(img, tl, 8, (0, 255, 0), -1)
#cv.circle(img, tr, 8, (0, 255, 0), -1)
#cv.circle(img, bl, 8, (0, 255, 0), -1)
#cv.circle(img, br, 8, (0, 255, 0), -1)  

cv.imshow('contour', img) 
k=cv.waitKey(0)
