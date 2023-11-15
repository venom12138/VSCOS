import cv2
import numpy as np
import glob
import os
import time
from sklearn.cluster import KMeans, DBSCAN
from PIL import Image
import matplotlib.pyplot as plt

def sparse_flow(flow, save_path, X=None, Y=None, stride=1, ):
    flow = flow.copy()
    flow[:,:,0] = -flow[:,:,0]
    if X is None:
        height, width, _ = flow.shape
        xx = np.arange(0,height,stride)
        yy = np.arange(0,width,stride)
        X, Y= np.meshgrid(xx,yy)
        X = X.flatten()
        Y = Y.flatten()

        
        sample_0 = flow[:, :, 0][xx]
        sample_0 = sample_0.T
        sample_x = sample_0[yy]
        sample_x = sample_x.T
        sample_1 = flow[:, :, 1][xx]
        sample_1 = sample_1.T
        sample_y = sample_1[yy]
        sample_y = sample_y.T

        sample_x = sample_x[:,:,np.newaxis]
        sample_y = sample_y[:,:,np.newaxis]
        new_flow = np.concatenate([sample_x, sample_y], axis=2)
    flow_x = new_flow[:, :, 0].flatten()
    flow_y = new_flow[:, :, 1].flatten()
    
    
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    
    ax.quiver(X,Y, flow_x, flow_y, color="
    ax.grid()
    
    plt.draw()
    plt.imsave(save_path, ax.get_figure().canvas.renderer.buffer_rgba())
    print(save_path )

flow_dir = './data/P04/flow_frames/P04_01'
u_dir = flow_dir + '/u'
v_dir = flow_dir + '/v'
count = 0

save_path = f'./data/P04/group/7991/sparse_flow/'
if not os.path.isdir(save_path):
    os.makedirs(save_path)
for u_img_path in sorted(list(glob.glob(u_dir + '/*.jpg'))):
    if count < 21224:
        count += 1
        continue
    else:
        count += 1
    

    v_img_path = u_img_path.replace('u','v')
    u_frame = np.expand_dims(cv2.imread(u_img_path, 0),axis=2)
    v_frame = np.expand_dims(cv2.imread(v_img_path, 0),axis=2)
    flow_frame = np.concatenate((u_frame,v_frame),axis=2).transpose(1,0,2) - 128
    
    sparse_flow(flow_frame, save_path = save_path+f'{count}.jpg', stride = 4)