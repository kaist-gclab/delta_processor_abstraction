import os
import argparse
import numpy as np
import util as ut
import volume_util as vt
from tqdm import tqdm
# import visualize as visu

## Visualize simplified mesh and corresponding segmentation result ##

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="prince_abs_1000")

args = parser.parse_args()

# Path
base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "datasets", args.data_dir)

gt_path = os.path.join(data_path, "gt_simp") # "gt_simp"
seg_res_path = os.path.join(data_path, "pseg")


meshes, names = ut.read_mesh(gt_path) # read all meshes / sorted
f_seg = ut.read_pseg_res(seg_res_path)

avg_abs = 0
cnt = 0
for i in tqdm(range(0,380)): # len(meshes)
    mesh = meshes[i]
    mesh = mesh.apply_scale(10.0)
    name = names[i]
    points = ut.get_vertex(mesh)
    faces = ut.get_face(mesh)
    seg = f_seg[i] # get related face segmentation
    cnt += 1
    part_meshes = vt.split_by_face_label(mesh, seg)

    # lst_points = []
    # lst_faces = []
    # lst_segs = []
    # lobb_points = []
    # lobb_faces = []
    # lobb_segs = []
    # laabb_points = []
    # laabb_faces = []
    # laabb_segs = []
    obb_part_meshes = []
    aabb_part_meshes = []
    
    for pkey in part_meshes.keys():
        part_mesh, part_vol = vt.fill_hole(part_meshes[pkey]) # convert into watertight mesh

        part_points = ut.get_vertex(part_mesh)
        part_faces = ut.get_face(part_mesh)
        part_seg = np.zeros((part_faces.shape[0],), dtype=np.int64)
        
        # lst_points.append(part_points)
        # lst_faces.append(part_faces)
        # lst_segs.append(part_seg)

        obb_box, obb_vol = vt.obb_without_outliers(part_points, 0.03) # 0.03
        aabb_box, aabb_vol = vt.aabb_without_outliers(part_points, 0.03)

        obb_mesh_box = vt.open3d_obb_to_trimesh_box(obb_box)
        aabb_mesh_box = vt.open3d_aabb_to_trimesh_box(aabb_box)
        obb_mesh, obb_vol = vt.inner_obb_from_obb(obb_mesh_box, 1.0, 0.9) # 0.8
        aabb_mesh, aabb_vol = vt.inner_obb_from_obb(aabb_mesh_box, 1.0, 0.9)

        obb_points = ut.get_vertex(obb_mesh).copy()
        obb_faces = ut.get_face(obb_mesh).copy()
        obb_seg = np.ones((obb_faces.shape[0],), dtype=np.int64)

        aabb_points = ut.get_vertex(aabb_mesh).copy()
        aabb_faces = ut.get_face(aabb_mesh).copy()
        aabb_seg = np.ones((aabb_faces.shape[0],), dtype=np.int64)
        
        # lobb_points.append(obb_points)
        # lobb_faces.append(obb_faces)
        # lobb_segs.append(obb_seg)

        # laabb_points.append(aabb_points)
        # laabb_faces.append(aabb_faces)
        # laabb_segs.append(aabb_seg)

        obb_part_meshes.append(obb_mesh)
        aabb_part_meshes.append(aabb_mesh)

    ioo, iou = vt.mesh_iou_solid(mesh, obb_part_meshes)
    aioo, aiou = vt.mesh_iou_solid(mesh, aabb_part_meshes)
    if iou > aiou:
        # print("Intersection over Orig {}: {:.3f}".format(i+1, ioo*100))
        avg_abs += ioo*100
    else:
        # print("Intersection over Orig {}: {:.3f}".format(i+1, aioo*100))
        avg_abs += aioo*100

    if cnt % 20 == 0:
        _class = cnt//20
        print("Intersection over Class {}: {:.3f}%".format(_class, avg_abs/20))
        avg_abs = 0
    
    
    