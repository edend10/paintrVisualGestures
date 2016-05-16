import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

#Eden Dolev | ed2566

#usage:
#Run main.py and an opencv window will open
#Need opencv3 module https://rivercitylabs.org/up-and-running-with-opencv3-and-python-3-anaconda-edition/
#Drag rectangle over marker color (should be small with distinct color that is unique in scene). Then click 'm' to toggle drawing mode with it.
#Click 'r' to start typing a shell command to associate with the drawing. Then 'enter' to save. Should have 'gestures' folder in the same folder as this script

#init bindings dict from pickle file if not empty
def initBindings():
    import pickle, os
    file_name = 'gestures/bindings.p'
    if not os.path.isfile(file_name):
        return dict()
    bindDict = dict()
    with open(file_name, 'rb') as f:
        bindDict = pickle.load(f)
    if not bindDict:
        return dict()
    return bindDict

#globals
bindings = initBindings()
ix, iy = -1,-1
drawing = False
save_vid = False
histogram_x1, histogram_y1, histogram_x2, histogramy2 = 0,0,0,0
hsv_lower = np.array([0,0,0], dtype=np.uint8)
hsv_upper = np.array([0,0,0], dtype=np.uint8)
typeMode = False
currentCommand = ''
message = 'Click and drag rectangle over desired marker color'
pythonPrint = print
timer_start = time.time()
timer_length = 5
rect_y = -1
rect_x = -1
rect_x_init = -1
rect_y_init = -1

#get rgb histogram for mouse selected square
def get_rgb_histogram(event, x, y, flags, param):
    global histogram_x1, histogram_y1, histogram_x2, histogram_y2, frame, hsv_upper, hsv_lower, rect_x, rect_y, rect_x_init, rect_y_init
    if rect_y != -1 and rect_x != -1:
        rect_y = y
        rect_x = x
    if event == cv2.EVENT_LBUTTONDOWN:
        histogram_x1 = x
        histogram_y1 = y
        rect_y = y
        rect_x = x
        rect_x_init = x
        rect_y_init = y
    if event == cv2.EVENT_LBUTTONUP:
        histogram_x2 = x
        histogram_y2 = y
        if histogram_x2 < histogram_x1:
            histogram_x2,histogram_x1 = histogram_x1,histogram_x2
        if histogram_y2 < histogram_y1:
            histogram_y2,histogram_y1 = histogram_y1,histogram_y2
        mask = np.zeros(frame.shape[:2], np.uint8)
        mask[histogram_y1:histogram_y2, histogram_x1:histogram_x2] = 255
        masked_img = cv2.bitwise_and(frame,frame,mask = mask)
        colors = ('b', 'g', 'r')
        bgrMax = list()
        for i,col in enumerate(colors):
            hist_mask = cv2.calcHist([frame], [i], mask, [256], [0,256])
            currColMax = hist_mask.argmax()
            bgrMax.append(currColMax)
        hsv = cv2.cvtColor(np.uint8([[bgrMax]]), cv2.COLOR_BGR2HSV)
        hMax = hsv[0][0][0]
        hsv_lower = np.array([hMax-10,100,100], dtype=np.uint8)
        hsv_upper = np.array([hMax+10, 255,255], dtype=np.uint8)
        print('New color marker set')
        rect_y = -1
        rect_x = -1
        

def print(msg):
    global message, timer_start
    pythonPrint(msg)
    message = msg
    timer_start = time.time()
    
#color code to bgr color
def get_color(color_code):
    if color_code == 0:
        return (255,255,255)
    elif color_code == 1:
        return (0,255,0)
    elif color_code == 2:
        return (0,0,255)

#save image file
def save_image(frame, gesture=False):
    global bindings, currentCommand
    import os
    num = -1
    folder_name = 'saved/images'
    if gesture:
        folder_name = 'gestures'
    for fn in os.listdir(folder_name):
        fn_arr = fn.split('.')
        if len(fn_arr) > 2:
            num_str = fn_arr[1]
            if fn_arr[0] == 'image' and fn_arr[2] == 'png':
                try:
                    if int(num_str) > num:
                        num = int(num_str)
                except ValueError:
                    pass
    num += 1
    file_name = 'image.%02d.png' % num
    if gesture:
        #command = input('Please enter a shell command:\n')
        command = currentCommand
        if command != '':
            import pickle

            bindings[file_name] = command
            cv2.imwrite('%s/%s' % (folder_name,file_name), frame)
            with open('%s/bindings.p' % folder_name, 'wb') as f:
                pickle.dump(bindings, f)
            print('Gesture recorded with command:"%s"' % command)
    else:
        print('Image saved: %s/%s' % (folder_name,file_name))
        cv2.imwrite('%s/%s' % (folder_name,file_name), frame)



#start new video file
def new_video():
    global save_vid, vid_out
    import os
    num = -1
    for fn in os.listdir('saved/videos'):
        fn_arr = fn.split('.')
        if len(fn_arr) > 2:
            num_str = fn_arr[1]
            if fn_arr[0] == 'video' and fn_arr[2] == 'mov':
                try:
                    if int(num_str) > num:
                        num = int(num_str)
                except ValueError:
                    pass
    num += 1
    fps = 30.0
    capSize = (1280,720)
    fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
    vid_out = cv2.VideoWriter('saved/videos/video.%02d.mov' % num,fourcc,fps,capSize,True) 
    save_vid = True

def nothing(x):
    pass

def detect_gesture(canvas):
    global bindings
    MATCH_THRESH = 10.0

    import os, sys
    imgray = cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY)
    if cv2.countNonZero(imgray) == 0:
        print('Empty canvas')
        return
    ret, thresh = cv2.threshold(imgray, 127,255,cv2.THRESH_BINARY) #127 is mid
    _,contours,_ = cv2.findContours(thresh,2,1)
    if not contours:
        print('Empty contours canvas')
        return
    cnt_compare = contours[0]
    best_match_val = sys.float_info.max
    best_match = 'No match'
    #matches = []
    for fn in os.listdir('gestures'):
        fn = 'gestures/%s' % fn
        curr_gray = cv2.imread(fn, 0)
        if cv2.countNonZero(curr_gray) == 0:
            #print('(empty - %s)' % fn)
            continue
        ret, thresh = cv2.threshold(curr_gray, 127,255,cv2.THRESH_BINARY) #127 is mid
        _,contours,_ = cv2.findContours(thresh,2,1)
        if not contours:
            #print('(empty contours - %s)' % fn)
            continue
        curr_cnt = contours[0]
        curr_match_val = cv2.matchShapes(cnt_compare, curr_cnt, 1, 0.0)
        if curr_match_val < best_match_val:
            best_match_val = curr_match_val
            best_match = fn.split('/')[-1]
    if best_match and best_match_val < MATCH_THRESH and bindings.get(best_match):
        command = bindings[best_match]
        print('Executing... %s | %s' % (command, os.popen(command).read()))
    else:
        print('No binding or matching gesture found')
    #for debugging:
        #matches.append((fn, curr_match_val))
    #print('Best match: %s with %f' % (best_match, best_match_val))
    #show_img = cv2.imread(best_match,0)
    #cv2.imshow('best match',show_img)
    #matches = sorted(matches, key=lambda x: x[1])
    #for fn, val in matches:
        #print('%s: %f' % (fn, val))




#new opencv window, capture, drawing canvas
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', get_rgb_histogram)
cap = cv2.VideoCapture(0)
canvas = np.zeros(cap.read()[1].shape, np.uint8)

#brush color trackbar
color_code = 0
#cv2.createTrackbar('Color', 'frame', 0, 2, nothing)
#cv2.createTrackbar('bottom_h', 'masked', 0, 180, nothing)
#cv2.createTrackbar('top_h', 'masked', 0, 180, nothing)

vid_out = None

frame = None
while(True):
    #frame by frame
    ret, frame = cap.read()

    #frame ops
    frame = cv2.flip(frame,1)
    #frame = cv2.GaussianBlur(frame,(5,5),0)
    frame = cv2.medianBlur(frame, 3)
    
    
    #bgr -> hsv
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

#    color_code = cv2.getTrackbarPos('Color', 'frame')
    color = get_color(color_code)

    #threshold hsv blue
    hsv_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    
    #apply color mask
    masked_frame = cv2.bitwise_and(frame, frame, mask=hsv_mask)

    #thresholding
    imgray = cv2.cvtColor(masked_frame,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 50,255,cv2.THRESH_BINARY) #127 is mid
#    thresh_gauss = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #erode+dilate*2
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    #contours for center of mass
    _,contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    width, height = frame.shape[:2]
    min_dist = 0.2
    if contours:
        M = cv2.moments(contours[0])
        #center of mass/centroid
        if M['m00']:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            #draw circle/line using centroid of color pointer
            if drawing:
                if ix == -1 and iy == -1:
                    cv2.circle(canvas,(cx,cy),2,color,-1)            
                elif abs(cx - ix) > min_dist * width or abs(cy - iy) > min_dist * height:
                    ix, iy = -1, -1
                else:
                    cv2.line(canvas,(ix,iy),(cx,cy),color,2)
                ix,iy = cx,cy
                


    #add drawing to frame
    new_frame = cv2.add(frame,canvas)
    cv2.putText(new_frame, 'LClick drag mouse to set marker color. "m" - toggle drawing mode. "r" - record shell-gesture binding. "d" - detect gesture and execute command. "p" save image', (20,height - 600) , cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
    if typeMode:
        cv2.putText(new_frame, 'Type shell command: ' + currentCommand, (20,30) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    elif time.time() - timer_start < timer_length and message != '':
        cv2.putText(new_frame, message, (20,30) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200)
    if rect_y != -1 and rect_x != -1:
        cv2.rectangle(new_frame, (rect_x_init, rect_y_init), (rect_x, rect_y), 255)


    #display frame in window
    cv2.imshow('frame', new_frame)

    if save_vid:
        vid_out.write(new_frame)

    #keys | q - quit | m - toggle drawing mode | c - clear screen | p - save picture | v - toggle video recording
    k = cv2.waitKey(1) & 0xFF
    if k == 13: #enter
        save_image(canvas, gesture=True)
        typeMode = False
    if not typeMode:
        if k == ord('q'):
            break
        elif k == ord('m'):
            print("DRAW MODE TOGGLE");
            drawing = not drawing
            ix,iy = -1,-1
        elif k == ord('c'):
            canvas = np.zeros(cap.read()[1].shape, np.uint8)
        elif k == ord('d'):
            detect_gesture(canvas)
            canvas = np.zeros(cap.read()[1].shape, np.uint8)
        elif k == ord('p'):
            save_image(new_frame)
        elif k == ord('r'):
            typeMode = True
            currentCommand = ''
    else:
        #mode for typing in shell commands
        if k != 255:
            if k == 127: #delete
                currentCommand = currentCommand[:-1]
            else:
                currentCommand += chr(k)
                


#release cv2 objects
cap.release()
vid_out.release()
cv2.destroyAllWindows()
