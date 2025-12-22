import os
import numpy as np
import trimesh

def read_mesh(dir_path, only_pref=False, ext=(".off", ".ply", ".obj")):
    """_summary_

    Args:
        dir_path (str): directory path where meshes are
        supporting mesh file extensions: .obj, .stl, .ply, .off

    Returns:
        list of meshes (trimesh.Trimesh): list of meshes
        Trimesh
        ├── vertices [n, 3]
        ├── faces [m, 3]
        ├── vertex_normals [n, 3]
        ├── face_normals [m, 3]
        ├── edges / edges_unique
        ├── is_watertight
        ├── volume / area / centroid
        ├── methods: show(), export(), fill_holes(), split()
    """
    fnames = [f for f in os.listdir(dir_path) if f.endswith(ext)]
    # count = len(os.listdir(dir_path))
    # fnames = ["{}.off".format(f) for f in range(1, count+1)]
    # _key = [int(f.split(".")[0]) for f in fnames]
    sfnames = sorted(fnames, key=lambda f: int(f.split(".")[0]))
    meshes = []
    for f in sfnames:
        fpath = os.path.join(dir_path, f)
        # print(fpath)
        mesh = trimesh.load(fpath)
        meshes.append(mesh)
    if only_pref:
        sfnames = ["".join([f.split(".")[0].split("_")[0], ".obj"]) for f in sfnames]

    return meshes, sfnames


def read_seg_res(dir_path):
    """Read segmentation result and seg(eseg)/sseg(seseg) file names
    Args:
        dir_path (python path): path where ground truth segmentation is saved
        layer (int, optional): nested file layer number. Defaults to 0.
    Returns:
        point_seg (python list): loaded segmentation in python list of lists. Inner component is ndarray
        sfile/sdir names: names of directory, files in sorted order
        seg_label_names (python list): doubly list of tags
    """
    filenames = [d for d in os.listdir(dir_path)]
    sfilenames = sorted(filenames, key=lambda d: int(d.split(".")[0].split("_")[0]))
    seg_files = ["{}.npz".format(f.split(".")[0]) for f in sfilenames]
    point_labels = []
    seg_label_names = []
    for elem in seg_files:
        fpath = os.path.join(dir_path, elem)
        part_label = np.load(fpath)
        part_label_tag = part_label.files
        point_seg = []
        seg_label_names.append(part_label_tag)
        for plabel_name in part_label_tag:
            point_seg.append(part_label[plabel_name])
        point_labels.append(point_seg)

    return point_labels, sfilenames, seg_label_names


def get_vertex(mesh, library="trimesh"):
    """_summary_: get vertex from mesh (read through different library)

    Args:
        mesh (_type_): original mesh
        library (str, optional): imported library of mesh. Defaults to "trimesh".

    Returns:
        vert (ndarray): (n, 3) sized vertex
    """
    if library == "trimesh":
        return mesh.vertices
    elif library == "o3d":
        return np.asarray(mesh.vertices)
    

def get_face(mesh, library="trimesh"):
    """_summary_: get face from mesh (read through different library)

    Args:
        mesh (_type_): original mesh
        library (str, optional): imported library of mesh. Defaults to "trimesh".

    Returns:
        face (ndarray): (n, 3) sized face
    """
    if library == "trimesh":
        return mesh.faces
    elif library == "o3d":
        return np.asarray(mesh.triangles)


def save_seseg(dirpath, fname, labels):
    fpath = os.path.join(dirpath, fname)
    num_classes = labels.shape[1]
    _format = " ".join(["%.6f"] * num_classes)
    np.savetxt(fpath, labels, fmt=_format, newline="\n")


def get_fnames(dir_path):
    filenames = [d for d in os.listdir(dir_path)]
    sfilenames = sorted(filenames, key=lambda d: int(d.split(".")[0].split("_")[0]))

    return sfilenames

def create_new_label(cur_label, map_dict):
    len_label = cur_label.shape[0] # get shape
    new_label = np.empty(len_label, dtype=np.int64)
    # get mask
    for _k in map_dict.keys():
        mask = (cur_label == _k)
        new_label[mask] = map_dict[_k]

    return new_label


def read_pseg(dirpath, fname):
    fpath = os.path.join(dirpath, fname)
    pseg = np.loadtxt(fpath, dtype=np.float32)

    return pseg


def save_pseg(dirpath, fname, labels):
    fpath = os.path.join(dirpath, fname)
    np.savetxt(fpath, labels, fmt="%d", newline="\n")


def read_pseg_res(dir_path):
    """_summary_: Read segmentation result and seg(eseg)/sseg(seseg) file names

    Args:
        dir_path (python path): path where ground truth segmentation is saved

    Returns:
        psegs (python list): loaded point segmentation in python list.
    """
    filenames = [d for d in os.listdir(dir_path)]
    sfilenames = sorted(filenames, key=lambda d: int(d.split(".")[0].split("_")[0]))
    psegs = []
    for fname in sfilenames:
        fpath = os.path.join(dir_path, fname)
        pseg = np.loadtxt(fpath, dtype=np.float32)
        psegs.append(pseg)

    return psegs