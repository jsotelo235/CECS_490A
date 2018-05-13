import  numpy as np
import  cv2
from  grabscreen import grab_screen
import time
    
def filter_WhiteYellow(image):
    """
    Filter the image, showing only a range of white and yellow
    """
    # Filter White
    threshold = 150 
    high_threshold = np.array([255, 255, 255]) #Bright white
    low_threshold = np.array([threshold, threshold, threshold]) #Soft White
    mask = cv2.inRange(image, low_threshold, high_threshold)
    white_img = cv2.bitwise_and(image, image, mask=mask)

    # Filter Yellow
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #Changing Color-space, HSV is better for object detection
    #For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]. 
    high_threshold = np.array([110,255,255]) #Bright Yellow
    low_threshold = np.array([50,50,50]) #Soft Yellow   
    mask = cv2.inRange(hsv_img, low_threshold, high_threshold)
    yellow_img = cv2.bitwise_and(image, image, mask=mask)

    # Combine the two above images
    filtered_img = cv2.addWeighted(white_img, 1., yellow_img, 1., 0.)

    return filtered_img


def change_perspective(img):
  img_size = (img.shape[1], img.shape[0])

  bot_width = .86
  mid_width = .08
  height_pct = .42
  bottom_trim = .935
  offset = img_size[0]*.25

  src = np.float32([[img.shape[1]*(.5 - mid_width/2), img.shape[0]*height_pct], [img.shape[1]*(.5 + mid_width/2), img.shape[0]*height_pct],\
   [img.shape[1]*(.5 + bot_width/2), img.shape[0]*bottom_trim], [img.shape[1]*(.5 - bot_width/2), img.shape[0]*bottom_trim]])
  dst = np.float32([[offset, 0], [img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]], [offset, img_size[1]]])
  # set fixed transforms based on image size

  # used to test that src points matched line
  # cv2.fillConvexPoly(img, src.astype('int32'), 1)
  # plt.imshow(img)
  # plt.title('lines')
  # plt.show()

  # create a transformation matrix based on the src and destination points
  M = cv2.getPerspectiveTransform(src, dst)

  #transform the image to birds eye view given the transform matrix
  warped = cv2.warpPerspective(img, M, (img_size[0], img_size[1]))
  return warped

def thresholding(image):
    """
    Apply Yellow and White Filter and create binary image
    """
    img_size = (image.shape[1], image.shape[0])
    #Filter white and Yellow to make it easier for more accurate Canny detection
    filtered_img = filter_WhiteYellow(image)
    #Convert image to gray scale
    gray = cv2.cvtColor(filtered_img, cv2.COLOR_RGB2GRAY)
    # Create binary based on detected pixels
    #binary_threshold = np.zeros_like(filtered_img)
    #binary_threshold[(gray > 0)] = 1
    # Warp the image
    #warped = cv2.warpPerspective(binary_threshold, M, img_size, flags=cv2.INTER_LINEAR)

    return gray

#------------------region_of_interest---------------
#
#   Filters out region that is not of interest 
#   Saves processing by only focusing on lane 
#
#---------------------------------------------------
#def region_of_interest(img, vertices):
#    mask = np.zeros_like(img)
#    cv2.fillPoly(mask, vertices, 255)
#    masked = cv2.bitwise_and(img, mask)
#    return masked  

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
'''
calculate the threshold of x or y sobel given certain thesh and kernel sizes
'''
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
  # grayscale image
  red = img[:, :, 0]

  # find abs sobel thresh
  if orient == 'x':
    sobel = cv2.Sobel(red, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
  else:
    sobel = cv2.Sobel(red, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
  
  #get abs value
  abs_sobel = np.absolute(sobel)
  scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))
  
  grad_binary = np.zeros_like(scaled)
  grad_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
  return grad_binary


'''
calculate magnitude of gradient given an image and threshold
'''
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
  # gray scale
  red = img[:, :, 0]
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  
  # given the magnitude of threshold for the combined two, return
  abs_x = np.absolute(cv2.Sobel(red, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
  abs_y = np.absolute(cv2.Sobel(red, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

  mag = np.sqrt(abs_x ** 2 + abs_y ** 2)
  scaled = (255*mag/np.max(mag))

  binary_output = np.zeros_like(scaled)
  binary_output[(scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])] = 1
  return binary_output

'''
calculate direction of gradient given image and thresh
'''
def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
  # red = img[:, :, 0]

  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  
  # given the magnitude of threshold for the combined two, return
  abs_x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
  abs_y = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

  sobel_dir = np.arctan2(abs_y, abs_x)

  binary_output = np.zeros_like(sobel_dir)
  binary_output[(sobel_dir >= thresh[0]) & (sobel_dir <= thresh[1])] = 1
  return binary_output

'''
calculate the threshold of the hls values
'''
def hls_thresh(img, thresh=(0, 255)):
  hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

  s_channel = hls[:, :, 2]

  binary_output = np.zeros_like(s_channel)
  binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1

  return binary_output

'''
get v channel from hsv
'''
def hsv_thresh(img, thresh=(0, 255)):
  hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

  v_channel = hsv[:, :, 2]

  binary_output = np.zeros_like(v_channel)
  binary_output[(v_channel > thresh[0]) & (v_channel <= thresh[1])] = 1

  return binary_output

'''
combine the thresholding functions
'''
def combo_thresh(img):


  x_thresholded = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(12, 120))

  y_thresholded = abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(25, 100))


  # was 90
  hls_thresholded = hls_thresh(img, thresh=(100, 255))

  hsv_thresholded = hsv_thresh(img, thresh=(50, 255))

  dir_thresholded = dir_thresh(img, sobel_kernel=15, thresh=(.7, 1.2))  
  

  mag_thresholded = mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100))



  binary_output = np.zeros_like(dir_thresholded)
  binary_output[((hsv_thresholded == 1) & (hls_thresholded == 1)) | ((x_thresholded == 1) & (y_thresholded == 1))] = 1

  # binary_output[(((dir_thresholded == 1) | (mag_thresholded == 1) ) & (hls_thresholded == 1)) | ((x_thresholded == 1) & (y_thresholded == 1))] = 1
  # 
  return binary_output

def process_img(original_image): 
    vertices = np.array([[10,500],[10,300],[300,200],[500,200],[800,300],[800,500]])
    processed_img = region_of_interest(original_image,[vertices])
    processed_img = combo_thresh(processed_img)
    #processed_img = filter_WhiteYellow(original_image)
    #processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    #processed_img = change_perspective(processed_img)
    #processed_img = cv2.GaussianBlur(processed_img,(5,5),0)

    ##edges 
    #lines = cv2.HoughLinesP(processed_img, 1, np.pi/180,180, np.array([]), 100, 5)
    #draw_lines(processed_img, lines)
    return processed_img


def main():
    last_time = time.time()
    while(True):
        screen = grab_screen(region=(0,40,800,640))    
        new_screen = process_img(screen)
        print('Loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        cv2.imshow('window', new_screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
main()

