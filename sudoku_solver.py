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

file_path = "D:/Dhruv/SUDOKU-SOLVER/sudoku-solver-app-cv/sudoku-solver-test.jpg"
img = cv.imread(file_path, 0) 
cv.imshow('display', img)
k=cv.waitKey(0)
#ratio = img.shape[0] / 300.0

#img = imutils.resize(img, height=300)
orig = img.copy()
orig2 = orig.copy()
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
cv.drawContours(img, [c], 0, (0, 255, 0), 3)
cv.imshow('display', img)
k=cv.waitKey(0)


#Finding extreme four points on a contour
maxSum = 0
minSum = 1000
maxDiff = -1000
minDiff = 1000
tr = np.array([0, 0])
tl = np.array([0, 0])
br = np.array([0, 0])
bl = np.array([0, 0])

for i in range(len(c)):
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

cv.circle(orig, tl, 8, (0, 255, 0), -1)
cv.circle(orig, tr, 8, (0, 255, 0), -1)
cv.circle(orig, bl, 8, (0, 255, 0), -1)
cv.circle(orig, br, 8, (0, 255, 0), -1)  


cv.imshow('display', orig)
#Function to transform the four points:

def four_point_transform(image, tl, tr, br, bl):
    rect = (tl, tr, br, bl)
    rect = np.float32(rect)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")

    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

warp_pic = four_point_transform(orig2, tl, tr, br, bl)

final_grid = cv.GaussianBlur(warp_pic, (5,5), cv.BORDER_DEFAULT) #Applying Gaussian blur on image to get rid of unnecessary noise 

final_grid = cv.adaptiveThreshold(final_grid, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 3)

final_grid = cv.bitwise_not(final_grid)

height = final_grid.shape[0]
width = final_grid.shape[1]

height_const = int(height/9)
width_const = int(width/9)

image_list = []

for i in range(1, height+height_const, height_const):
    for j in range(5, width+width_const, width_const):
        image_list.append(final_grid[i:i+height_const, j:j+width_const])

cv.imshow('display', image_list[20])
k=cv.waitKey(0)


for i in image_list:
    cnts = cv.findContours(i, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv.contourArea)
    cv.drawContours(i, [c], 0, (0, 255, 0), 3)

for i in range(len(image_list)):
    if image_list[i].shape[0]>0 and image_list[i].shape[1]>0 and i!=0:
        cv.imshow('display', image_list[i])
        k=cv.waitKey(0)
        

cv.destroyAllWindows()
