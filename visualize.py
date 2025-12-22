import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def vis_mult_seg(lst_points, lst_faces, lst_labels):
    """_summary_: visualize multiple cuboids

    Args:
        lst_points (python list of ndarray): (ncub, 8)
        lst_faces (python list of ndarray): (ncub, 6)
        lst_labels (python list of ndarray): (ncub, 6)
    """
    vis_lst = []
    for i in range(len(lst_points)):
        cur_points = lst_points[i]
        cur_faces = lst_faces[i]
        cur_labels = lst_labels[i]
        o3d_mesh = get_face_color_abs(cur_labels, cur_points, cur_faces) # vert, face, color already included
        o3d_line = get_edge(cur_points, cur_faces)
        vis_lst.append(o3d_mesh)
        vis_lst.append(o3d_line)
 
    o3d.visualization.draw_geometries(vis_lst)


def vis_face_seg(points, faces, labels):
    """_summary_: visualize segmentation

    Args:
        points (ndarray): points (nump, 3)
        labels (ndarray): initial patch labels (nump, )
    """
    o3d_mesh = get_face_color_mesh(labels, points, faces) # vert, face, color already included
    o3d_line = get_edge(points, faces)
 
    o3d.visualization.draw_geometries([o3d_mesh, o3d_line])


def get_face_color_mesh(labels, points, faces):
    """_summary_: This get labels of FACE, points, face as input
                    and returns colored mesh according to the label

    Args:
        labels (ndarray): labels (numf, 3)
        points (ndarray): points (nump, 3)
        faces (ndarray): faces (numf, 3)
        X cmap (str, optional): Color map, removed now. Defaults to 'tab20'.

    Returns:
        out_mesh: Open3d mesh that contains points, faces, 
    """
    # color_map = plt.get_cmap(cmap) # clen, 3
    new_vert = points[faces].reshape(-1, 3) # (M*3, 3)
    num_faces = faces.shape[0]
    new_tri = np.arange(num_faces*3, dtype=np.int32).reshape(num_faces, 3)
    # fcolors = face_colors_from_labels(labels, cmap)
    fcolors = face_colors_from_custom_labels(labels)
    new_colors = np.repeat(fcolors, 3, axis=0)

    out_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(new_vert),
        triangles=o3d.utility.Vector3iVector(new_tri)
    )
    out_mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)
    
    return out_mesh


# Not used
def face_colors_from_labels(face_labels, cmap_name="tab20"):
    face_labels = np.asarray(face_labels)
    classes, inv = np.unique(face_labels, return_inverse=True)  # stable mapping
    K = len(classes)

    color_map = plt.get_cmap(cmap_name)
    # Table of K distinct RGB colors sampled from colormap
    table = color_map(np.linspace(0, 1, max(K, 1)))[:, :3]  # drop alpha
    return table[inv].astype(np.float64)


def face_colors_from_custom_labels(face_labels):
    face_labels = np.asarray(face_labels)
    classes, inv = np.unique(face_labels, return_inverse=True)  # stable mapping
    # K = len(classes)
    palette = np.array([
        [1,   0,   0],  # 0 -> red
        [  0,   0, 1],  # 1 -> blue
        [  0, 200/255,   0],  # 2 -> green
        [1, 140/255,   0],  # 3 -> orange
        [148/255,   0, 211/255],  # 4 -> purple
        [  0, 200/255, 200/255],  # 5 -> cyan
        [1, 215/255,   0],  # 6 -> yellow
        [204/255, 121/255, 167/255],  # 7 -> magenta (Okabe–Ito #CC79A7)
        [96/255, 78/255, 42/255], # 8 -> brown
        [0, 158/225, 115/225], # 9 -> teal
        [ 86/255, 180/255, 233/255],  # 10 -> sky blue   (#56B4E9)
        [153/255, 153/255,  51/255],  # 11 -> olive      (#999933)
        [136/255,  34/255,  85/255],  # 12 -> wine       (#882255)
        [ 51/255,  34/255, 136/255],  # 13 -> dark blue  (#332288)
        [127/255, 127/255, 127/255],  # 14 -> gray       (#7F7F7F)
        [204/255, 102/255, 119/255],  # 15 -> rose       (#CC6677)
        [ 88/255, 170/255, 108/255],  # 16 -> medium green (#58AA6C)
        [255/255, 255/255, 255/255],  # 17 -> white      (#708090)
        [184/255, 134/255,  11/255],  # 18 -> ochre (황토색) (#B8860B)
        [245/255, 186/255,  187/255],  # 19 -> light pink (#F5BABB)
        [183/255, 163/255,  227/255],  # 20 -> light violet (#B7A3E3)
        [0,   0,   0],  # 21 -> black
        [0,   0,   0],  # 22 -> black
        [0,   0,   0],  # 23 -> black
        [0,   0,   0],  # 24 -> black
        [0,   0,   0],  # 25 -> black
        [0,   0,   0],  # 26 -> black
        [0,   0,   0],  # 27 -> black
        [0,   0,   0],  # 28 -> black
        [0,   0,   0],  # 29 -> black
        [0,   0,   0],  # 30 -> black
        [0,   0,   0],  # 31 -> black
        [0,   0,   0],  # 32 -> black
        [0,   0,   0],  # 33 -> black
        [0,   0,   0],  # 34 -> black
        [0,   0,   0],  # 35 -> black
        [0,   0,   0],  # 36 -> black
        [0,   0,   0],  # 37 -> black
        [0,   0,   0],  # 38 -> black
        [0,   0,   0],  # 39 -> black
        [0,   0,   0],  # 40 -> black
        [0,   0,   0],  # 41 -> black
        [0,   0,   0],  # 42 -> black
        [0,   0,   0],  # 43 -> black
        [0,   0,   0],  # 44 -> black
        [0,   0,   0],  # 45 -> black
        [0,   0,   0],  # 46 -> black
    ], dtype=np.float32)
    # Table of K distinct RGB colors sampled from colormap
    colors = palette[face_labels]
    return colors.astype(np.float64)


def get_edge(points, faces):
    """_summary_

    Args:
        points (ndarray): (nump, 3)
        faces (ndarray): (numf, 3)

    Returns:
        line_seg (ndarray): calculated all edges segment coordinates (numf*3, 2)
    """
    # create line_set
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points) # 이름 같지만 되긴함

    vidx1 = faces[:, 0]
    vidx2 = faces[:, 1]
    vidx3 = faces[:, 2]

    eidx1 = np.stack([vidx1, vidx2], axis=-1)
    eidx2 = np.stack([vidx2, vidx3], axis=-1)
    eidx3 = np.stack([vidx3, vidx1], axis=-1)

    edge = np.concatenate([eidx1, eidx2, eidx3], axis=0)
    line_set.lines = o3d.utility.Vector2iVector(edge)
    ledge = edge.shape[0]
    line_set.colors = o3d.utility.Vector3dVector(np.tile([[0, 0, 0]], (ledge, 1)))

    return line_set


#-----(not used functions below)-----
def vis_init_patch(points, faces, labels):
    """_summary_

    Args:
        points (ndarray): points (nump, 3)
        labels (ndarray): initial patch labels (nump, )
    """
    o3d_mesh = get_mesh(points, faces)
    colors = get_vert_color(labels)
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d_line = get_edge(points, faces)
 
    o3d.visualization.draw_geometries([o3d_mesh, o3d_line])


# ---------- only used in mesh_abs_visualize ----------
def get_face_color_abs(labels, points, faces):
    """_summary_: This get labels of FACE, points, face of one cuboid
                    and returns colored cuboid according to the label

    Args:
        labels (ndarray): labels (numf, 3)
        points (ndarray): points (nump, 3)
        faces (ndarray): faces (numf, 3)

    Returns:
        out_mesh: Open3d mesh that contains points, faces, 
    """
    new_vert = points[faces].reshape(-1, 3) # (M*3, 3)
    num_faces = faces.shape[0]
    new_tri = np.arange(num_faces*3, dtype=np.int32).reshape(num_faces, 3)
    fcolors = face_colors_for_abs(labels)
    new_colors = np.repeat(fcolors, 3, axis=0)

    out_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(new_vert),
        triangles=o3d.utility.Vector3iVector(new_tri)
    )
    out_mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)
    
    return out_mesh


def face_colors_for_abs(face_labels):
    """_summary_

    Args:
        face_labels (python list of ndarray): (ncub, 6)

    Returns:
        colors: color for all face labels
    """
    face_labels = np.asarray(face_labels)
    classes, inv = np.unique(face_labels, return_inverse=True)  # stable mapping
    K = len(classes)
    palette = np.array([
        [183/255, 163/255,  227/255], # 0 -> light violet
        [ 86/255, 180/255, 233/255],  # 1 -> sky blue   (#56B4E9)
        [  0,   0, 1],  # 1 -> blue
        [  0, 200/255,   0],  # 2 -> green
        [1, 140/255,   0],  # 3 -> orange
        [148/255,   0, 211/255],  # 4 -> purple
        [  0, 200/255, 200/255],  # 5 -> cyan
        [1, 215/255,   0],  # 6 -> yellow
        [204/255, 121/255, 167/255],  # 7 -> magenta (Okabe–Ito #CC79A7)
        [96/255, 78/255, 42/255], # 8 -> brown
        [0, 158/225, 115/225], # 9 -> teal
        [ 86/255, 180/255, 233/255],  # 10 -> sky blue   (#56B4E9)
        [153/255, 153/255,  51/255],  # 11 -> olive      (#999933)
        [136/255,  34/255,  85/255],  # 12 -> wine       (#882255)
        [ 51/255,  34/255, 136/255],  # 13 -> dark blue  (#332288)
        [127/255, 127/255, 127/255],  # 14 -> gray       (#7F7F7F)
        [204/255, 102/255, 119/255],  # 15 -> rose       (#CC6677)
        [ 88/255, 170/255, 108/255],  # 16 -> medium green (#58AA6C)
        [255/255, 255/255, 255/255],  # 17 -> white      (#708090)
        [184/255, 134/255,  11/255],  # 18 -> ochre (황토색) (#B8860B)
        [245/255, 186/255,  187/255],  # 19 -> light pink (#F5BABB)
        [183/255, 163/255,  227/255],  # 20 -> light violet (#B7A3E3)
    ], dtype=np.float32)
    # Table of K distinct RGB colors sampled from colormap
    colors = palette[face_labels]
    return colors.astype(np.float64)