import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def vis_mult_seg(lst_points, lst_faces, lst_labels):
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
    """_summary_

    Args:
        points (ndarray): points (nump, 3)
        labels (ndarray): initial patch labels (nump, )
    """
    o3d_mesh = get_face_color_mesh(labels, points, faces) # vert, face, color already included
    o3d_line = get_edge(points, faces)
 
    o3d.visualization.draw_geometries([o3d_mesh, o3d_line])


def get_face_color_mesh(labels, points, faces, cmap='tab20'):
    color_map = plt.get_cmap(cmap) # clen, 3
    new_vert = points[faces].reshape(-1, 3) # (M*3, 3)
    num_faces = faces.shape[0]
    new_tri = np.arange(num_faces*3, dtype=np.int32).reshape(num_faces, 3)
    # highlight = zero_out_except(labels, 0, off_value=1, remap_kept=True, start_at=2)
    # fcolors = face_colors_from_labels(labels, cmap)
    fcolors = face_colors_from_custom_labels(labels)
    new_colors = np.repeat(fcolors, 3, axis=0)

    out_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(new_vert),
        triangles=o3d.utility.Vector3iVector(new_tri)
    )
    out_mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)
    
    return out_mesh


def zero_out_except(face_labels, keep, off_value=0, remap_kept=False, start_at=1):
    """
    face_labels: (F,) ints
    keep: int or iterable of ints (labels to keep)
    off_value: background value for all other labels (0 or -1 recommended)
    remap_kept: if True, remap kept labels to [start_at .. start_at+K-1]
    start_at: first id for remapped kept labels (default 1 so 0 can stay background)
    """
    labels = np.asarray(face_labels)
    keep = np.atleast_1d(keep)

    mask = np.isin(labels, keep)
    out = np.full_like(labels, off_value)

    if not remap_kept:
        # keep original ids for kept labels
        out[mask] = labels[mask]
        return out

    # remap only the kept labels to a compact range
    kept_vals = np.unique(labels[mask])
    # deterministic order
    kept_vals.sort()
    # build mapping: kept_val -> start_at + i
    remap = {v: start_at + i for i, v in enumerate(kept_vals)}

    # apply remap only on the kept positions
    out_kept = np.empty(mask.sum(), dtype=out.dtype)
    for i, v in enumerate(labels[mask]):
        out_kept[i] = remap[v]
    out[mask] = out_kept
    return out


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
    K = len(classes)
    # palette = np.array([
    #     [228,  26,  28],  # 0 -> red         (#E41A1C)
    #     [ 55, 126, 184],  # 1 -> blue        (#377EB8)
    #     [ 77, 175,  74],  # 2 -> green       (#4DAF4A)
    #     [152,  78, 163],  # 3 -> purple      (#984EA3)
    #     [230, 159,   0],  # 4 -> orange      (#E69F00, Okabe–Ito)
    #     [  0, 158, 115],  # 5 -> teal (greenish blue)       (#009E73, Okabe–Ito)
    #     [240, 228,  66],  # 6 -> yellow      (#F0E442, Okabe–Ito)
    # ], dtype=np.uint8)
    palette = np.array([
        # [0,   0,   0],  # 0 -> red
        # [0,   0,   0],  # 0 -> red
        # [0,   0,   0],  # 0 -> red
        # [0,   0,   0],  # 0 -> red
        # [0,   0,   0],  # 0 -> red
        # [0,   0,   0],  # 0 -> red
        # [0,   0,   0],  # 0 -> red
        # [0,   0,   0],  # 0 -> red
        # [0,   0,   0],  # 0 -> red
        # [0,   0,   0],  # 0 -> red
        # [0,   0,   0],  # 0 -> red
        # [0,   0,   0],  # 0 -> red
        # [0,   0,   0],  # 0 -> red
        # [0,   0,   0],  # 0 -> red
        # [0,   0,   0],  # 0 -> red
        # [0,   0,   0],  # 0 -> red
        # [0,   0,   0],  # 0 -> red
        # [0,   0,   0],  # 0 -> red
        # [0,   0,   0],  # 0 -> red
        # [0,   0,   0],  # 0 -> red
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
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red
        [0,   0,   0],  # 0 -> red

        # [1,   0,   0],  # 0 -> red
        # [  0,   0, 1],  # 1 -> blue
        # [  0, 200/255,   0],  # 2 -> green
        # [1, 140/255,   0],  # 3 -> orange
        # [148/255,   0, 211/255],  # 4 -> purple
        # [  0, 200/255, 200/255],  # 5 -> cyan
        # [1, 215/255,   0],  # 6 -> yellow
        # [204/255, 121/255, 167/255],  # 7 -> magenta (Okabe–Ito #CC79A7)
        # [96/255, 78/255, 42/255], # 8 -> brown
        # [0, 158/225, 115/225], # 9 -> teal
        # [ 86/255, 180/255, 233/255],  # 10 -> sky blue   (#56B4E9)
        # [153/255, 153/255,  51/255],  # 11 -> olive      (#999933)
        # [136/255,  34/255,  85/255],  # 12 -> wine       (#882255)
        # [ 51/255,  34/255, 136/255],  # 13 -> dark blue  (#332288)
        # [127/255, 127/255, 127/255],  # 14 -> gray       (#7F7F7F)
    ], dtype=np.float32)
    # Table of K distinct RGB colors sampled from colormap
    colors = palette[face_labels]
    return colors.astype(np.float64)


def get_edge(points, faces):
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


def get_mesh(points, faces):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(points)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    return o3d_mesh


def get_vert_color(labels, cmap='tab20'):
    label_norm = labels.astype(float) / max(labels.max(), 1)
    color_map = plt.get_cmap(cmap)
    colors = color_map(label_norm)[:, :3]

    return colors


# ---------- only used in mesh_abs_visualize ----------
def get_face_color_abs(labels, points, faces):
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