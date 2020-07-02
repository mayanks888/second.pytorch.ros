

point_indices = box_np_ops.nuscenes_points_in_rbbox(points, rbbox_lidar,annos['rotation_matrix'])
gt_points = points[point_indices[:, i]]



def corner_to_surfaces_3d(corners):
    """convert 3d box corners from corner function above
    to surfaces that normal vectors all direct to internal.

    Args:
        corners (float array, [N, 8, 3]): 3d box corners.
    Returns:
        surfaces (float array, [N, 6, 4, 3]):
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    surfaces = np.array([
        [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
        [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]],
        [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
        [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]],
        [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]],
        [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]],
    ]).transpose([2, 0, 1, 3])
    return surfaces

def points_in_convex_polygon_3d_jit(points,
                                    polygon_surfaces,
                                    num_surfaces=None):
    """check points is in 3d convex polygons.
    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces
            a polygon contain
    Returns:
        [num_points, num_polygon] bool array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d_jit(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = points[i, 0] * normal_vec[j, k, 0] \
                     + points[i, 1] * normal_vec[j, k, 1] \
                     + points[i, 2] * normal_vec[j, k, 2] + d[j, k]
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret

def nuscenes_points_in_rbbox(points, rbbox,rotation_matrix):
    location = rbbox[:, :3]
    dimension = rbbox[:, 3:6]
    ry = rbbox[:, 6]
    iNumObj = ry.shape[0]
    corners_3d_list=[]
    for iIndex in range(iNumObj):
        w,l,h = dimension[iIndex,:]
        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        rot_mat = rotation_matrix[iIndex,:,:]
        corners = np.vstack((x_corners, y_corners, z_corners))
        corners = np.dot(rot_mat, corners)
        centre = location[iIndex,:]
        centre = np.expand_dims(centre, axis=1)
        rbboxes = centre  # np.dot(box.orientation.rotation_matrix, centre)
        x, y, z = location[iIndex,:]
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z
        corners_3d = corners.T
        corners_3d = np.expand_dims(corners_3d, axis=0)
        corners_3d_list.append(corners_3d)

    corner_3d = corners_3d_list[0]
    for iIndex in range(iNumObj-1):
        array2 = corners_3d_list[iIndex+1]
        corner_3d = np.concatenate((corner_3d,array2),axis=0)
    surfaces = corner_to_surfaces_3d(corner_3d)
    indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return indices