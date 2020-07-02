import mayavi.mlab as mlab
import os
import numpy as np
import matplotlib.pyplot as plt
def read_calib_file(filepath):
    ''' Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    '''
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def cart2hom(pts_3d):
    ''' Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    '''
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom


def project_velo_to_ref(pts_3d_velo):
    pts_3d_velo = cart2hom(pts_3d_velo)  # nx4
    return np.dot(pts_3d_velo, np.transpose(V2C))


def project_ref_to_velo(pts_3d_ref):
    pts_3d_ref = cart2hom(pts_3d_ref)  # nx4
    return np.dot(pts_3d_ref, np.transpose(C2V))


def project_rect_to_ref(pts_3d_rect):
    ''' Input and Output are nx3 points '''
    # t1=np.linalg.inv(R0)
    # t2=pts_3d_rect
    # t3=np.dot(t1,t2)
    # t4 = np.transpose(t3)
    return np.transpose(np.dot(np.linalg.inv(R0), np.transpose(pts_3d_rect)))


def project_ref_to_rect(pts_3d_ref):
    ''' Input and Output are nx3 points '''
    return np.transpose(np.dot(R0, np.transpose(pts_3d_ref)))


def project_rect_to_velo(pts_3d_rect):
    ''' Input: nx3 points in rect camera coord.
        Output: nx3 points in velodyne coord.
    '''
    pts_3d_ref = project_rect_to_ref(pts_3d_rect)
    return project_ref_to_velo(pts_3d_ref)


def project_velo_to_rect(pts_3d_velo):
    pts_3d_ref = h.project_velo_to_ref(pts_3d_velo)
    return project_ref_to_rect(pts_3d_ref)

def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def transform_from_rot_trans(R, t):
    ''' Transforation matrix from rotation matrix and translation vector. '''
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr

def pointCloudToBirdsEyeView(ax2, velo, bb3d):
    ax2.set_xlim (-10,10)
    ax2.set_ylim (-5,35)
    hmax = velo[:,2].max()
    hmin = velo[:,2].min()
    hmean = velo[:, 2].mean()
    hmeadian = np.median ( velo[:, 2] )
    hstd = np.std(velo[:, 2])
    #print ('scalledh', hmax, hmean, hmeadian, hmin, hstd, scalledh.shape, scalledh[:10])
    norm = colors.Normalize(hmean-2*hstd, hmean+2*hstd, clip=True)
    sc2= ax2.scatter(-velo[:,1],
             velo[:,0],
             s = 1,
             c=velo[:,2],
             cmap = 'viridis',
             norm=norm,
             marker = ".",
             )
    ax2.scatter(-bb3d[:,1],
             bb3d[:,0],
             c='red')
    ax2.set_facecolor('xkcd:grey')
    plt.colorbar(sc2)
    plt.waitforbuttonpress()

def draw_lidar_simple(pc, color=None):
    ''' Draw lidar points. simplest set up. '''
    fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[:,2]
    #draw points
    mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color, color=None, mode='point', colormap = 'gnuplot', scale_factor=1, figure=fig)
    #draw origin
    mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.5)
    #draw axis
    axes=np.array([
        [2.,0.,0.,0.],
        [0.,2.,0.,0.],
        [0.,0.,2.,0.],
    ],dtype=np.float64)
    mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)
    mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    mlab.show()
    return fig

def draw_gt_boxes3d(gt_boxes3d,iIndex,fig,color=(1,1,1), line_width=1, draw_text=True, text_scale=(1,1,1), color_list=None):
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        if draw_text: mlab.text3d(b[4,0], b[4,1], b[4,2], '%d'%iIndex, scale=text_scale, color=color, figure=fig)
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
    # mlab.show()
    mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig

def hv_in_range(x, y, z, fov, fov_type='h'):
    """
    Extract filtered in-range velodyne coordinates based on azimuth & elevation angle limit
    Args:
    `x`:velodyne points x array
    `y`:velodyne points y array
    `z`:velodyne points z array
    `fov`:a two element list, e.g.[-45,45]
    `fov_type`:the fov type, could be `h` or 'v',defualt in `h`
    Return:
    `cond`:condition of points within fov or not
    Raise:
    `NameError`:"fov type must be set between 'h' and 'v' "
    """
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if fov_type == 'h':
        return np.logical_and(np.arctan2(y, x) > (-fov[1] * np.pi/180), np.arctan2(y, x) < (-fov[0] * np.pi/180))
    elif fov_type == 'v':
        return np.logical_and(np.arctan2(z, d) < (fov[1] * np.pi / 180), np.arctan2(z, d) > (fov[0] * np.pi / 180))
    else:
        raise NameError("fov type must be set between 'h' and 'v' ")


def boxes_in_3d(boxes_lidar,fig):
    for iIndex, box3d in enumerate(boxes_lidar, start=0):
        if iIndex>500:
            break
        location = box3d[:3]
        dimensn = box3d[3:6]
        angle = box3d[-1]

        h = dimensn[2]  # box height
        w = dimensn[1]  # box width
        l = dimensn[0]  # box length (in meters)
        t = (location[0], location[1], location[2])
        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))
        ry = angle
        R = rotz(ry)
        corners = np.dot(R, corners)
        x = t[0]
        y = t[1]
        z = t[2]
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z
        corners_3d = corners.T
        color = (0, 1, 0)
        line_width = 1
        mlab.points3d(x, y, z, color=(1, 1, 1), mode='sphere', scale_factor=0.4)
        # mlab.text3d(x, y, z, class_names[labels_lidar[iIndex]], scale=(.5, .5, .5))
        # mlab.text3d(x + 1, y + 1, z + 1, str(round(scores_lidar[iIndex], 2)), scale=(.5, .5, .5))

        # mlab.title('car')
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            mlab.plot3d([corners_3d[i, 0], corners_3d[j, 0]], [corners_3d[i, 1], corners_3d[j, 1]],
                        [corners_3d[i, 2], corners_3d[j, 2]], color=color, tube_radius=None, line_width=line_width,
                        figure=fig)
            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([corners_3d[i, 0], corners_3d[j, 0]], [corners_3d[i, 1], corners_3d[j, 1]],
                        [corners_3d[i, 2], corners_3d[j, 2]], color=color, tube_radius=None, line_width=line_width,
                        figure=fig)
            i, j = k, k + 4
            mlab.plot3d([corners_3d[i, 0], corners_3d[j, 0]], [corners_3d[i, 1], corners_3d[j, 1]],
                        [corners_3d[i, 2], corners_3d[j, 2]], color=color, tube_radius=None, line_width=line_width,
                        figure=fig)
    mlab.show()