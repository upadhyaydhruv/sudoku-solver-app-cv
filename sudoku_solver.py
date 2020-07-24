import numpy as np
import cv2 as cv
import sys

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


file_path = "sudoku-solver-test.jpg"
img = cv.imread(file_path, 0) 
if img is None:
    sys.exit("Could not read the image.")

gauss_blurred = cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT) #Applying Gaussian blur on image to get rid of unnecessary noise 

adaptive_threshold = cv.adaptiveThreshold(gauss_blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 3) #

adaptive_threshold = cv.bitwise_not(adaptive_threshold)

kernel = np.ones((3,3),np.uint8)
dilated = cv.dilate(adaptive_threshold, kernel, iterations = 1)

cv.imshow('display', dilated)
k=cv.waitKey(0)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY_INV)
image, contours, hierarchy = cv.findContours(thresh_gray, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

mx = (0,0,0,0)
mx_area = 0

for cont in contours:
    x,y,w,h = cv.boundingRect(cont)
    area = w*h
    if area>mx_area:
        mx = x,y,w,h
        mx_area = area
x,y,w,h = mx

roi = img[y:y+h, x:x+w]

cv.imwrite('cropped_sudoku.jpg', roi)
cv2.rectangle(img,(x,y),(x+w,y+h),(200,0,0),2)
cv2.imwrite('Image_cont.jpg', img)
 
cropped='cropped_sudoku.jpg'

cv.imshow('display', cropped)
#cv2.imshow('image', image)
k=cv.waitKey(0)

#Finding largest blob (the sudoku square)

'''
cnts = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv.contourArea, reverse=True)
for c in cnts:
    # Highlight largest contour
    cv.drawContours(dilated, [c], -1, (36,35,36), 3)
    break
'''

#Finding contours on original image
'''
rand, contours, h = cv.findContours(img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv.contourArea, reverse=True)
polygon = contours[0]

bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in
                      polygon]), key=operator.itemgetter(1))
top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in
                  polygon]), key=operator.itemgetter(1))
bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in
                     polygon]), key=operator.itemgetter(1))
top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in
                   polygon]), key=operator.itemgetter(1))

list_of_points = [polygon[top_left][0], polygon[top_right][0], polygon[bottom_left][0], polygon[bottom_right][0]]
src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32') 

print(list_of_points[0])


side = max([  distance_between(bottom_right, top_right), 
            distance_between(top_left, bottom_left),
            distance_between(bottom_right, bottom_left),   
            distance_between(top_left, top_right) ])
'''

