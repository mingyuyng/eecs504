# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 12:27:59 2018

@author: nadha
"""
# F18 EECS 504 HW1p1 Homography Estimation
import numpy as np
import matplotlib.pyplot as plt
import os

import eta.core.image as etai

def get_correspondences(img1, img2, n):
    '''
    Function to pick corresponding points from two images and save as .npy file
    Args:
	img1: Input image 1
	img1: Input image 2
	n   : Number of corresponding points 
   '''
    
    correspondence_pts = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    coords = []
    def onclick(event):
        global ix, iy
        ix, iy = event.xdata, event.ydata
        print("The current point is: ")
        print (ix, iy)
        
        coords.append((ix, iy))

        if len(coords) == n:
            fig.canvas.mpl_disconnect(cid)
            plt.close()
        return coords

    ax.imshow(img1)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    correspondence_pts.append(coords)
    coords = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img2)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    correspondence_pts.append(coords)
    
    np.save('football_pts_'+str(n)+'.npy',correspondence_pts)

def main(n):
    '''
    This function will find the homography matrix and then use it to find corresponding marker in football image 2
    '''
    # reading the images
    img1 = etai.read('football1.jpg')
    img2 = etai.read('football2.jpg')

    filepath = 'football_pts_'+str(n)+'.npy'
    # get n corresponding points
    if not os.path.exists(filepath):
        get_correspondences(img1, img2,n)
    
    correspondence_pts = np.load(filepath)
    
    XY1 = correspondence_pts[0]
    XY2 = correspondence_pts[1]
    # plotting the Fooball image 1 with marker 33
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(img1)
    u=[1210,1701]
    v=[126,939]
    ax.plot(u, v,color='yellow')
    ax.set(title='Football image 1')
    #plt.show()
    
    #------------------------------------------------
    # FILL YOUR CODE HERE
    # Your code should estimate the homogrphy and draw the  
    # corresponding yellow line in the second image.

    # Form the A and y for least square problem: min||Ax - y||
    A = []
    y = []
    for i in range(n):
        tmp = np.kron(np.eye(2),np.append(XY1[i,:],1))
        tmp = np.concatenate((tmp,-np.outer(XY2[i,:],XY1[i,:])),axis=1)
        if i == 0:
            A = tmp
            y = XY2[i,:]
        else:
            A = np.append(A,tmp,axis=0)
            y = np.append(y,XY2[i,:],axis=0)
    
    # Solve the least square problem and regenerate the matrix H
    x  = np.linalg.inv(np.dot(A.T,A)).dot(A.T).dot(y)
    x  = np.append(x,1) 
    H  = np.reshape(x,(3,3))
    
    # Generate the new starting and ending points of the yellow line
    T  = np.append(u,v)
    T  = np.append(T,np.ones(2))
    T  = np.reshape(T,(3,2))
    T_new = np.dot(H,T)
    T_new = T_new/T_new[-1,:]
    
    # Plot image 2 and the corresponding yellow line
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(img2)
    u1 = T_new[0,:]
    v1 = T_new[1,:]
    ax.plot(u1, v1,color='yellow')
    ax.set(title='Football image 2')
    plt.show()


if __name__ == "__main__": 
    
    #------------------------------------------------
    # FILL BLANK HERE
    # Specify the number of pairs of points you need.
    n = 10
    #------------------------------------------------
    main(n)
