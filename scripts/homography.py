import cv2
import numpy as np


def justBoard(im_src):
    size = (200,300,3)
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
                        [407,161],
                        [2771, 113],
                        [3005, 3983],
                        [215, 3947]
                        ], dtype=float
                       )
    # Calculate the homography
    h, status = cv2.findHomography(pts_src, pts_dst)
    # Warp source image to destination
    im_dst = cv2.warpPerspective(im_src, h, size[0:2])
    return im_dst

def homo(file):
    #Read image
    im_src = cv2.imread(file)
    #Collect four corners of board [do this by inspection]
    pts_src = np.array([[141, 131], [480, 159], [493, 630],[64, 601]])
 
    # Read destination image.
    im_dst = cv2.imread('flat_board.png')
    pts_dst = np.array(
                       [
                        [0,0],
                        [size[0] - 1, 0],
                        [size[0] - 1, size[1] -1],
                        [0, size[1] - 1 ]
                        ], dtype=float
                       )
    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)
    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
    return im_out

def print_img(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name,1080, 720)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

MAX_MATCHES = 500
GOOD_MATCH_PERCENT = 0.15


def alignImages(im1, im2):

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
  
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_MATCHES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
  
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
  
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)
  
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
  
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
  
    return im1Reg, h

if __name__ == '__main__':
    # im =cv2.imread('table.jpg')
    # x = justBoard(im)
    # cv2.imwrite('justtable.jpg', x)
    # print_img('img', x)


    im = cv2.imread('different_state.png')
    ref = cv2.imread('justtable.jpg')
    #dst = justBoard(im)
    print_img('img', ref)

    im1 = cv2.imread('table.png')
    print_img('img',im1)

    #im2 = cv2.imread('different_state.png', cv2.IMREAD_COLOR)
    imReg, h = alignImages(im1, ref)
    print_img('img', imReg)
    print(h)





