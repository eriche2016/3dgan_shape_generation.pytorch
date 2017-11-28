import os 
import pickle 
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 

def getVoxelFromMat(path, cube_len=64):
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
    return voxels


def getVFByMarchingCubes(voxels, threshold=0.5):
    v, f = sk.marching_cubes_classic(voxels, level=threshold)
    return v, f


def SavePloat_Voxels(voxels, path, iteration, threshold=0.1):
    voxels_scores = voxels.copy() 
    voxels = voxels[:8].__ge__(threshold) # G with tanh, 0, for G with sigmoid, 0.5 
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='green')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
    plt.savefig(path + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.close()

    with open(path + '/{}.pkl'.format(str(iteration).zfill(3)), "wb") as f:
        # protocol =2 so that we can load it from python2 in my Window machine 
        pickle.dump(voxels_scores, f, protocol=2)# protocol = pickle.HIGHEST_PROTOCOL)


# do gradient clip 
def clip_gradient(optimizer, grad_clip):
    assert grad_clip>0, 'gradient clip value must be greater than 1'
    for group in optimizer.param_groups:
        for param in group['params']:
            # gradient
            param.grad.data.clamp_(-grad_clip, grad_clip)

def plotFromVoxels(voxels):
    z, x, y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='red')
    plt.show()


def plotVoxelVisdom(voxels, visdom, title):
    v, f = getVFByMarchingCubes(voxels)
    visdom.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title))


def SavePloat_Voxels2offline(voxels, path, threshold=0.1):
    voxels = voxels[:8].__ge__(threshold) # G with tanh, 0, for G with sigmoid, 0.5 
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='green')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')

    plt.savefig(path + '/{}.png'.format(str('_0_' + str(threshold*10))), bbox_inches='tight')
    plt.close()


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    import numpy as np
    from math import factorial
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def render_graphs(save_dir,epoch, track_d_loss_iter, track_d_loss, track_recon_loss_iter = None, track_recon_loss=None, track_valid_loss_iter=None, track_valid_loss=None): 
    if not os.path.exists(save_dir+'/plots/'):
        os.makedirs(save_dir+'/plots/')
    if track_recon_loss is not None:
        if len(track_recon_loss)>51: 
            smoothed_recon = savitzky_golay(track_recon_loss, 51, 3)
            plt.plot(track_recon_loss_iter, track_recon_loss,color='blue') 
            plt.plot(track_recon_loss_iter,smoothed_recon , color = 'red')
            if track_valid_loss is not None:
                plt.plot(track_valid_loss_iter, track_valid_loss ,color='green')
            plt.savefig(save_dir+'/plots/recon_' + str(epoch)+'.png' )
            plt.clf()
    if len(track_d_loss)> 301: 
        smoothed_d_loss = savitzky_golay(track_d_loss, 301, 3)
        plt.plot(track_d_loss_iter, track_d_loss)
        plt.plot(track_d_loss_iter, smoothed_d_loss, color = 'red')
        plt.savefig(save_dir+'/plots/' + str(epoch)+'.png' )
        plt.clf()

def save_values(save_dir,track_d_loss_iter, track_d_loss, track_recon_loss_iter = None, track_recon_loss=None, track_valid_loss_iter=None, track_valid_loss=None):
    np.save(save_dir+'/plots/track_d_loss_iter', track_d_loss_iter)
    np.save(save_dir+'/plots/track_d_loss', track_d_loss)
    if track_recon_loss is not None:
        np.save(save_dir+'/plots/track_recon_loss_iter', track_recon_loss_iter)
        np.save(save_dir+'/plots/track_recon_loss', track_recon_loss)
    if track_valid_loss is not None: 
        np.save(save_dir+'/plots/track_valid_loss_iter', track_valid_loss_iter)
        np.save(save_dir+'/plots/track_valid_loss', track_valid_loss)
