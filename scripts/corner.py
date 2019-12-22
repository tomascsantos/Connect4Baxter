import numpy as np
import cv2


def justBoard(im_src, one, two, three, four):
    size = (300,200,3)
    im_dst = np.zeros(size, np.uint8)
    pts_dst = np.array(
                       [
                        [0,0],
                        [size[0] - 1, 0],
                        [size[0] - 1, size[1] -1],
                        [0, size[1] - 1 ]
                        ], dtype=float
                       )
    # Show image and wait for 4 clicks.
    cv2.imshow("Image", im_src)
    pts_src = np.array(
                       [
                        [one[0],one[1]],
                        [two[0], two[1]],
                        [three[0], three[1]],
                        [four[0], four[1]]
                        ], dtype=float
                       )
    # Calculate the homography
    h, status = cv2.findHomography(pts_src, pts_dst)
    # Warp source image to destination
    im_dst = cv2.warpPerspective(im_src, h, size[0:2])
    return im_dst

def corners(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,8,17,0.02)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.02*dst.max()]=[0,0,255]
    #img[dst<=0.04*dst.max()]=[0,0,0]
    return img

def corners2(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,8,17,0.02)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.03*dst.max()]=[0,0,255]
    return dst>0.03*dst.max()

def read_image(img_name, grayscale=False):
    if not grayscale:
        img = cv2.imread(img_name)
    else:
        img = cv2.imread(img_name, 0)
    return img

def print_img(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name,1080, 720)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def threshold(img_name, color):
    img = cv2.imread(img_name)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_r1 = np.array([0,70,50])
    upper_r1 = np.array([10,255,255])
    lower_r2 = np.array([160,70,50])
    upper_r2 = np.array([179,255,255])

    lower_b = np.array([100,30,30])
    upper_b = np.array([150,255,255])

    lower_y = np.array([5,30,30])
    upper_y = np.array([40,255,255])

    lower_p = np.array([150,80,120])
    upper_p = np.array([176,180,255])

    lower_gg = np.array([40,0,0])
    upper_gg = np.array([100,255,255])

    red1 = cv2.inRange(img_hsv,lower_r1,upper_r1)
    red2 = cv2.inRange(img_hsv,lower_r2,upper_r2)
    red = red1 + red2

    blue = cv2.inRange(img_hsv,lower_b,upper_b)
    yellow = cv2.inRange(img_hsv,lower_y,upper_y)
    pink = cv2.inRange(img_hsv,lower_p,upper_p)
    green2 = cv2.inRange(img_hsv,lower_gg,upper_gg)
    #print_img('red',red)
    #print_img('blue',blue)
    #print_img('yellow',yellow)
    #print_img('default',img)
    if color == 'BLUE':
        return blue
    if color == 'RED':
        return red
    if color == 'YELLOW':
        return yellow
    if color == 'PINK':
        return pink
    if color == 'GREEN2':
        return green2

    return img_hsv

def findCorners(file):
    #1 RGB -> GRAY  (cvCvtColor)
    img = cv2.imread(file)
    #2 Smooth(cvSmooth)
    #3 Threshold (cvThreshold)
    #4 Detect Edges (cvCanny)
    #5 Find Contours (cvFindContours)
    #6 Approximate contours with linear features (cvApproxPoly)
    #7 Find "rectangles" which were structures that: had polygonalized contours 
    #  possessing 4 points, were of sufficient area, had adjacent edges were ~90 degrees, 
    #  had distance between "opposite" vertices was of sufficient size, etc.
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares

def filter_homo(mask):
    indices = np.where(mask != 0)
    coordinates = list(zip(indices[0], indices[1]))

def filter_hc(mask):
    #1 find top corner
    #2 left or right?
    #3 elimate all other points within 200
    #4 find other top corner using min
    #5 find bottom corner using max y
    #6 interpolate using  top corners?

    indices = np.where(mask != 0)
    coord_xfirst = list(zip(indices[0], indices[1]))
    coord_yfirst = list(zip(indices[1], indices[0]))

    xs = indices[0]
    ys = indices[1]

    #1
    top_corn = min(coord_yfirst) #rev
    #2
    if top_corn[1] < 575:
        top_left = [top_corn[1], top_corn[0]] #reg
    else:
        top_right = [top_corn[1], top_corn[0]] #reg
    #3
    temp = [top_corn[1], top_corn[0]] #reg
    for c in coord_xfirst:
        if within(temp, c, 200):
            coord_xfirst.remove(c)
    return coord_xfirst

def within(ref_pt, check_pt, dist):
    d = np.sqrt((ref_pt[0] - check_pt[0])**2 + (ref_pt[1] - check_pt[1])**2)
    if d < dist:
        return True
    else:
        return False


def getBoardtoHomo(img):
    im = cv2.imread(img)
    im = cv2.imread(img)
    mask = threshold(img, 'PINK')

    #Corners
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)
        # calculate x,y coordinate of center
        if M["m00"] != 0 and M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        points.append((cX, cY))

    corners = []
    corners.append(min(points, key=lambda x: x[0] + x[1]))
    corners.append(max(points, key=lambda x: x[0] + x[1]))
    corners.append(max(points, key=lambda x: x[0] - x[1]))
    corners.append(max(points, key=lambda x: x[1] - x[0]))
    for c in corners:
        cv2.circle(im, (c[0], c[1]), 5, (255, 255, 255), -1)
        cv2.putText(im, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    #Get Colors
    x = justBoard(im, corners[0], corners[2], corners[1], corners[3])
    img_hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    lower_r1 = np.array([0,70,50])
    upper_r1 = np.array([10,255,255])
    lower_r2 = np.array([160,70,50])
    upper_r2 = np.array([179,255,255])
    lower_y = np.array([5,30,30])
    upper_y = np.array([40,255,255])
    yellow = cv2.inRange(img_hsv,lower_y,upper_y)
    red1 = cv2.inRange(img_hsv,lower_r1,upper_r1)
    red2 = cv2.inRange(img_hsv,lower_r2,upper_r2)
    red = red1 + red2 

    kernel = np.ones((4,4),np.uint8)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, kernel)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel)

    #BoardState
    current_pos_row = 0
    current_pos_col = 0
    output = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]

    height, width = red.shape
    start_x = 0
    start_y = 0
    y_inc = height / 6
    x_inc = width / 7

    for i in range(42):
        end_x = int(start_x + x_inc)
        if end_x > width - 1:
            end_x = width - 1
        end_y = int(start_y + y_inc)
        if end_y > height - 1:
            end_y  = height - 1
        if len(np.where(yellow[start_y:end_y, start_x: end_x] != 0)[0]) > 200:
            output[current_pos_row][current_pos_col] = 1 #YELLOW = 1
        if len(np.where(red[start_y:end_y, start_x: end_x] != 0)[0]) > 200:
            output[current_pos_row][current_pos_col] = 2 #RED = 2
        
        current_pos_col += 1
        start_x = int(start_x + x_inc)
        if current_pos_col > 6:
            current_pos_row += 1
            current_pos_col = 0
            start_y = int(start_y + y_inc)
            start_x = 0

    for i in range(6):
        print(output[i])

    return output

if __name__ == '__main__':
    #kernel = np.ones((10,10),np.uint8)
    #mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #print_img('img', mask2)
    #res = cv2.bitwise_and(im,im, mask= mask2)
    #out = corners(res)

    img = 'board_hand_pink.png'
    x = getBoardtoHomo(img)
    im = cv2.imread(img)
    #_____FOR HOMO
    im = cv2.imread(img)
    mask = threshold(img, 'PINK')


    #_____SQUARES
    # im = cv2.imread(img)
    # squares = find_squares(im)
    # print(squares)
    # x = cv2.drawContours(im, squares, -1, (0, 255, 0), 3 )
    # print_img('img', x)


    #_______Harris Corners
    # im = cv2.imread(img)
    # print_img("img", im)
    # mask = threshold(img, 'BLUE')
    # print_img('mask1', mask)

    # kernel = np.ones((10,10),np.uint8)
    # mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # print_img('mask2',mask2)

    # res = cv2.bitwise_and(im,im, mask= mask2)
    # print_img('img', res)
    # out = corners(res)
    # print_img("img",out)

    #________Board State
    # img = 'ref.jpg'
    # mask = threshold(img, 'YELLOW')
    # print_img('yel', mask)
    # kernel = np.ones((3,3),np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # print_img('mask', mask)
    # numLabels, labelImage, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)

    # mask_red = threshold(img,'RED')
    # mask_yel  = threshold(img, 'YELLOW')
    # print(im.shape)
    # print_img('red', mask_red)
    # print_img('yel', mask_yel)
    # out = board_state(img)
    # print_img('i', mask_yel[171: 228, 50:100])
    # print(np.any(mask_yel[50:100, 171: 228] != 0))
    # print(out)


    #________TABLE
    # im = cv2.imread(img)
    # mask = threshold(img,'GREEN2')
    # #print_img('mask', mask)
    # kernel = np.ones((6,6),np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # res = cv2.bitwise_and(im,im, mask= mask)

    # out = corners(res)

    # lower_r = np.array([0,80,80])
    # upper_r = np.array([10,255,255])
    # img_hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    # thresh = cv2.inRange(img_hsv,lower_r,upper_r)

    # print_img("thresh", thresh)

    # contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # points = []
    # for c in contours:
    #     # calculate moments for each contour
    #     M = cv2.moments(c)
    #     # calculate x,y coordinate of center
    #     cX = int(M["m10"] / M["m00"])
    #     cY = int(M["m01"] / M["m00"])
    #     points.append((cX, cY))


    # corners = []
    # corners.append(min(points, key=lambda x: x[0] + x[1]))
    # corners.append(max(points, key=lambda x: x[0] + x[1]))
    # corners.append(max(points, key=lambda x: x[0] - x[1]))
    # corners.append(max(points, key=lambda x: x[1] - x[0]))


    # print(corners)
    # for c in corners:
    #     cv2.circle(im, (c[0], c[1]), 5, (255, 255, 255), -1)
    #     cv2.putText(im, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # print_img("thresh", im)

    # x = justBoard(im, corners[0], corners[2], corners[1], corners[3])
    # print_img('x', x)
    # print(x.shape)














