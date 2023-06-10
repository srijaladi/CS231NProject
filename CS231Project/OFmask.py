import numpy as np
import torch
import torch.nn as nn
#import cv2
import os
from PIL import Image as im
from matplotlib import pyplot as plt
import skimage
from skimage.measure import block_reduce as AVG_POOL
import OpticalFlow
from OpticalFlow import optical_flow, optical_flow_pair
import Models
from Models import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

TRUE_C, TRUE_H, TRUE_W= 3, 128, 128 
IN_C, IN_H, IN_W = 3, 32, 32
IN_DIM = (IN_C, IN_H, IN_W)
FLOW_DIM = (2, IN_H, IN_W)
TRUE_DIM = (TRUE_C, TRUE_H, TRUE_W)
BATCH_TEST_DIM = (1, IN_C, IN_H, IN_W)
BATCH_FLOW_DIM = (1, *FLOW_DIM)

POOL_METHOD = np.mean
POOL_FACTOR = TRUE_H//IN_H
assert TRUE_W//IN_W == POOL_FACTOR

DATA_FOLDER  = "DataFolder/"
FRAME_FOLDER = DATA_FOLDER + "Frames/"
LOG_FOLDER   = DATA_FOLDER + "Logging/"
EXAMPLES_FOLDER = DATA_FOLDER + "Examples/"
CURR_VIDEO_NAME = "test_video"
CURR_VIDEO_FRAMES_NAME = CURR_VIDEO_NAME + "_frames"
LOAD_FRAME_DATA_VERBOSE = True

TRAIN_FRAME_DIFF = 2
TEST_FRAME_DIFF = TRAIN_FRAME_DIFF//2
TRAIN_ITERATIONS = 10000
BATCH_SIZE = 64
VAL_SIZE = 500
ACC_CUTOFF = 100

BATCH_DIM = (BATCH_SIZE, *IN_DIM)

def model_sanity_check(model):
    x1 = torch.zeros(*BATCH_TEST_DIM).to(DEVICE)
    x2 = torch.zeros(*BATCH_TEST_DIM).to(DEVICE)
    x3 = torch.zeros(*BATCH_FLOW_DIM).to(DEVICE)
    model(x1, x2, x3)
    return True

def load_frame_data(folderpath, verbose = False):
    frames_list = sorted(os.listdir(folderpath))
    num_frames  = len(frames_list)
    all_frames  = np.zeros((num_frames, *IN_DIM))
    for idx, frame in enumerate(frames_list):
        filepath = folderpath + "/" + str(frame)
        curr_frame = np.loadtxt(filepath)
        curr_frame = np.reshape(curr_frame, TRUE_DIM)
        for c in range(TRUE_C):
            all_frames[idx][c] = AVG_POOL(curr_frame[c], POOL_FACTOR, POOL_METHOD)
        if verbose and (idx+1)%100 == 0:
            print("Retrieved " + str(idx+1) + " frames so far")
    if verbose: print("Got all frames length: " + str(len(all_frames)))
    all_frames = all_frames[800:-800]
    if verbose: print("Stripped beginning and ending of frames")
    return torch.from_numpy(all_frames).float().to(DEVICE)

def gen_batch_sample(data, flow_data, batch_size, frame_diff : int):
    l = int(len(data))
    b_idxs = torch.randint(0,l-frame_diff-frame_diff, size = (batch_size,))
    a_idxs = b_idxs + frame_diff + frame_diff
    t_idxs = b_idxs + frame_diff
    
    before_batch = data[b_idxs]
    after_batch  = data[a_idxs]
    target_batch = data[t_idxs]
    flow_batch   = flow_data[b_idxs]
    
    return (before_batch, after_batch, flow_batch, target_batch)

def gen_batch_pure(data, flow_data, frame_diff: int):
    l = int(len(data))
    b_idxs = torch.arange(l - frame_diff - frame_diff).long()
    a_idxs = b_idxs + frame_diff + frame_diff
    t_idxs = b_idxs + frame_diff
    
    before_batch = data[b_idxs]
    after_batch  = data[a_idxs]
    target_batch = data[t_idxs]
    flow_batch   = flow_data[b_idxs]
    
    return (before_batch, after_batch, flow_batch, target_batch)

def gen_flow_data(data, frame_diff):
    np_data = data.cpu().detach().numpy()
    amt = np_data.shape[0] - frame_diff - frame_diff
    res = np.zeros((amt, 2, *(np_data[0][0].shape)))
    for i in range(amt - frame_diff - frame_diff):
        res[i] = np.copy(optical_flow_pair(np_data[i], np_data[i + frame_diff + frame_diff]))
    return torch.from_numpy(res).float().to(DEVICE)

def log_vals(vals : dict):
    for key, val in vals.items():
        filepath = LOG_FOLDER + key + ".txt"
        np.savetxt(filepath, np.array(val))
        
def log_models(fl, cl, gl):
    torch.save(fl.state_dict(), LOG_FOLDER + "features_model_params.txt")
    torch.save(cl.state_dict(), LOG_FOLDER + "concatention_model_params.txt")
    torch.save(gl.state_dict(), LOG_FOLDER + "generation_model_params.txt")
    
def log_model(model):
    torch.save(model.state_dict(), LOG_FOLDER + "model_params.txt")
        
def conv_image(image):
    np_img = np.swapaxes(np.swapaxes(image.cpu().detach().numpy(), 0, 2), 0, 1)
    return np_img

def save_image(filepath, image):
    np_img = conv_image(image)
    plt.clf()
    plt.imshow(np_img/255, interpolation='nearest')
    plt.savefig(filepath)
    plt.clf()
        
def save_images(folderpath, input_images, output_image):
    if not(os.path.exists(folderpath)): os.mkdir(folderpath)
    before_image, after_image, target_image = input_images
    
    save_image(folderpath + "/before.png", before_image)
    save_image(folderpath + "/after.png",  after_image)
    save_image(folderpath + "/target.png", target_image)
    save_image(folderpath + "/output.png",  output_image)
    
def save_examples(mode, model, batch):
    assert mode == "val" or mode == "test" or mode == "train" or mode == "train_val"
    loss_per = model.eval_loss_per(*batch)

    idxs = torch.argsort(loss_per)

    for i in range(5):
        idx = idxs[i]
        x_1, x_2, x_3, t_x = batch[0][idx], batch[1][idx], batch[2][idx], batch[3][idx]
        out_x = model(x_1, x_2, x_3)[0]
        folderpath = EXAMPLES_FOLDER + mode + "_best_" + str(i+1)
        save_images(folderpath, (x_1, x_2, t_x), out_x)
        
    for i in range(1,6):
        idx = idxs[-i]
        x_1, x_2, x_3, t_x = batch[0][idx], batch[1][idx], batch[2][idx], batch[3][idx]
        out_x = model(x_1, x_2, x_3)[0]
        folderpath = EXAMPLES_FOLDER + mode + "_worst_" + str(i)
        save_images(folderpath, (x_1, x_2, t_x), out_x)
        

k_size = (5,5)
in_h, in_w, feature_size = IN_H, IN_W, 32

"""
f_l = [torch.nn.Conv2d(8, 32, k_size, padding = 'same'), torch.nn.MaxPool2d((2,2), stride = 2), torch.nn.BatchNorm2d(32), torch.nn.ReLU(), 
      torch.nn.Conv2d(32,16, k_size, padding = 'same'), torch.nn.BatchNorm2d(16), torch.nn.ReLU(),
      torch.nn.Conv2d(16, 8, k_size, padding = 'same'), torch.nn.MaxPool2d((2,2), stride = 2), torch.nn.BatchNorm2d(8), torch.nn.ReLU(),
      torch.nn.Conv2d(8, 4, k_size, padding = 'same'), torch.nn.BatchNorm2d(4), torch.nn.ReLU(),
      torch.nn.Conv2d(4, 2, k_size, padding = 'same'), torch.nn.MaxPool2d((2,2), stride = 2), torch.nn.BatchNorm2d(2), torch.nn.ReLU(),
      torch.nn.Conv2d(2, 1, k_size, padding = 'same'), torch.nn.BatchNorm2d(1), torch.nn.ReLU(),
      torch.nn.Flatten(), torch.nn.Linear(16 * 16, IN_C * IN_H * IN_W)]
"""
f_l = [torch.nn.Conv2d(8, 32, k_size, padding = 'same'), torch.nn.MaxPool2d((2,2), stride = 2), torch.nn.ReLU(), 
      torch.nn.Conv2d(32, 16, k_size, padding = 'same'), torch.nn.MaxPool2d((2,2), stride = 2), torch.nn.ReLU(),
      torch.nn.Flatten(), torch.nn.Linear(16 * IN_H//4 * IN_W//4, IN_C * IN_H * IN_W)]
g_l = [torch.nn.Unflatten(1, (3,IN_H,IN_W))]
m_l = [torch.nn.Conv2d(2, 1, k_size, padding = 'same'), torch.nn.Sigmoid()]

learning_rates = {'features':1e-3, 'generation':1e-3}

FrameModel = AdvancedFlowFrameGenModel(in_h, in_w, f_l, g_l, m_l, torch.nn.functional.mse_loss, learning_rates).to(DEVICE)
model_sanity_check(FrameModel)

print("MODEL CREATED")

baseline_video_data = data = load_frame_data(FRAME_FOLDER + CURR_VIDEO_FRAMES_NAME, verbose = LOAD_FRAME_DATA_VERBOSE)

print("LOADED DATA")

train_data = data

print("COMPUTING OPTICAL FLOW")

train_flow_data = flow_data = gen_flow_data(data, TRAIN_FRAME_DIFF)

print("FINISHED OPTICAL FLOW COMPUTATION AND STORING")

in_train_batch = gen_batch_sample(data, flow_data, VAL_SIZE, TRAIN_FRAME_DIFF)
val_batch  = gen_batch_sample(data, flow_data, VAL_SIZE, TEST_FRAME_DIFF)
test_batch = gen_batch_sample(data, flow_data, VAL_SIZE, TEST_FRAME_DIFF)

print("CREATED TESTING AND VAL BATCHES")

log = {'train_losses' : [], 'val_losses' : [], 'val_accuracies' : []}
    
for itr in range(TRAIN_ITERATIONS):
    train_batch = gen_batch_sample(data, flow_data, BATCH_SIZE, TRAIN_FRAME_DIFF)
    train_loss  = FrameModel.backprop(*train_batch)
    val_acc     = FrameModel.eval_acc(*val_batch, ACC_CUTOFF)
    val_loss    = FrameModel.eval_loss_total(*val_batch)
    
    log['train_losses'].append(train_loss)
    log['val_losses'].append(val_loss)
    log['val_accuracies'].append(val_acc)
    
    log_vals(log)
    
    if itr%10 == 0:
        print("ITERATION: " + str(itr))
        print("TRAIN LOSS: " + str(train_loss))
        print("VAL ACCURACY: " + str(val_acc))
        print("VAL LOSS: " + str(val_loss))
        save_examples("val", FrameModel, val_batch)
        save_examples("train_val", FrameModel, in_train_batch)
        log_model(FrameModel)


test_acc = FrameModel.eval_acc(*test_batch, ACC_CUTOFF)
test_loss = FrameModel.eval_loss_total(*test_batch)

print("LOSS ON TEST SET: " + str(test_loss))
print("ACCURACY ON TEST SET: " + str(test_acc))

save_examples("test", FrameModel, test_batch)






