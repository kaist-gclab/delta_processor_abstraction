import os
import argparse
import numpy as np
import util as ut
import volume_util as vt
import visualize as visu

## Visualize simplified mesh and corresponding segmentation result ##

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="prince_simp_1000") # "prince_ben", "prince_simp_1000"
parser.add_argument("--save_dir", type=str, default="prince_abs_1000")

args = parser.parse_args()

# Path
base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "datasets", args.data_dir)
save_path = os.path.join(base_path, "datasets", args.save_dir, "pseg")

gt_path = os.path.join(data_path, "gt_conn") # "gt_simp"
seg_res_path = os.path.join(data_path, "seg_conn") # "seg_simp"

# make point segmentation saving directory
os.makedirs(save_path, exist_ok=True)

meshes, names = ut.read_mesh(gt_path) # read all meshes / sorted
point_seg, eseg_name, seseg_name = ut.read_seg_res(seg_res_path)

for i in range(360,380): # len(meshes)
    mesh = meshes[i]
    mesh = mesh.apply_scale(10.0)
    name = names[i]
    points = ut.get_vertex(mesh)
    faces = ut.get_face(mesh)
    cur_seg = point_seg[i] # get related face segmentation
    num_lst = [1, 2, 1, 0, 4, 7, 0, 4, 9, 5, 8, 6, 0, 8, 1, 1, 7, 3, 3, 3] # human
    num_lst = [2, 0, 0, 11, 7, 2, 6, 12, 6, 10, 6, 6, 3, 3, 8, 4, 0, 11, 7, 1] # cup
    num_lst = [2, 0, 0, 7, 3, 1, 7, 0, 5, 2, 1, 5, 0, 1, 0, 1, 2, 4, 1, 0] # glasses
    num_lst = [9, 7, 1, 1, 6, 6, 6, 7, 1, 8, 5, 2, 6, 6, 6, 2, 4, 1, 4, 6] # airplane
    num_lst = [6, 4, 1, 9, 3,  2, 3, 2, 3, 3,  3, 1, 1, 0, 6,  0, 6, 4, 8, 3] # ants
    num_lst = [0, 5, 6, 0, 2,  0, 2, 2, 0, 0,  5, 4, 0, 1, 0,  4, 0, 0, 0, 0] # chair
    num_lst = [0, 4, 6, 2, 2,  4, 0, 7, 0, 1,  5, 4, 2, 4, 2,  1, 7, 12, 8, 5] # octopus
    num_lst = [5, 1, 11, 3, 2,  3, 5, 2, 4, 9,  1, 4, 4, 5, 9,  5, 12, 10, 4, 2] # table
    num_lst = [3, 6, 4, 5, 7, 10, 2, 1, 11, 11, 9, 3, 2, 4, 7, 9, 10, 10, 6, 8] # teddy bear
    num_lst = [2, 0, 3, 5, 4,  1, 0, 7, 4, 6,  5, 7, 6, 4, 2,  2, 0, 2, 0, 9] # hand
    num_lst = [5, 4, 7, 3, 4, 3, 6, 1, 4, 8, 5, 2, 1, 5, 1, 6, 1, 9, 3, 7] # piler
    num_lst = [1, 10, 8, 3, 4,  0, 3, 3, 6, 1,  6, 7, 5, 4, 0,  4, 6, 2, 0, 0] # fish
    num_lst = [2, 2, 4, 4, 10, 3, 0, 5, 0, 1, 9, 3, 4, 9, 0, 1, 3, 4, 3, 1] # bird
    num_lst = [2, 3, 1, 2, 7, 0, 2, 2, 0, 2, 3, 1, 1, 2, 0, 0, 2, 6, 5, 1] # armadillo
    num_lst = [7,0,2,7,1, 4,0,7,4,0, 4,0,2,2,7, 4,4,0,6,6] # bust
    num_lst = [0,2,2,2,1, 0,0,9,0,1, 0,2,2,0,1, 7,0,0,2,0] # mech
    num_lst = [0,2,0,3,0, 0,0,6,3,5, 0,0,0,0,0, 1,0,6,4,2] # bearing
    num_lst = [1,2,4,5,1, 10,0,10,5,9, 5,1,10,10,5, 7,3,7,6,3] # vase
    num_lst = [0,7,8,9,1, 3,2,3,4,3, 3,6,9,1,2, 4,0,5,3,12] # fourleg
    elem = num_lst[i%20] # elem = idx of cur_seg
    seg = cur_seg[elem] # face segmentation
    assert faces.shape[0] == cur_seg[elem].shape[0], "len vertices and len labels not same"
    # cur_dict = lst_dict[i%20]
    # new_seg = ut.create_new_label(seg, cur_dict) # convert seg using dictionary
    # if i == 377:
    #     seg2 = cur_seg[3]
    #     cur_dict2 = {7:2, 6:2, 10:100, 9:100, 8:100, 0:100, 3:100, 1:100, 14:100, 2:100, 4:100, 13:100, 12:100, 11:100, 5:100}
    #     new_seg2 = ut.create_new_label(seg2, cur_dict2)
    #     new_seg = np.minimum(new_seg, new_seg2)
    # part_meshes = vt.split_by_face_label(mesh, new_seg)
    part_meshes = vt.split_by_face_label(mesh, seg)

    tobb_vol = 0
    tpart_vol = 0
    lst_points = []
    lst_faces = []
    lst_segs = []
    lobb_points = []
    lobb_faces = []
    lobb_segs = []
    laabb_points = []
    laabb_faces = []
    laabb_segs = []
    obb_part_meshes = []
    aabb_part_meshes = []
    for pkey in part_meshes.keys():
        part_mesh, part_vol = vt.fill_hole(part_meshes[pkey]) # convert into watertight mesh

        part_points = ut.get_vertex(part_mesh)
        part_faces = ut.get_face(part_mesh)
        part_seg = np.zeros((part_faces.shape[0],), dtype=np.int64)
        # visu.vis_face_seg(part_points, part_faces, part_seg) # visualize part_mesh
        lst_points.append(part_points)
        lst_faces.append(part_faces)
        lst_segs.append(part_seg)

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
        
        lobb_points.append(obb_points)
        lobb_faces.append(obb_faces)
        lobb_segs.append(obb_seg)

        laabb_points.append(aabb_points)
        laabb_faces.append(aabb_faces)
        laabb_segs.append(aabb_seg)

        # print("Part Area Difference: {:.3f}".format(obb_vol - part_vol))
        # print("Obb Area: {:.3f}".format(obb_vol))

        tobb_vol += obb_vol
        tpart_vol += part_vol

        obb_part_meshes.append(obb_mesh)
        aabb_part_meshes.append(aabb_mesh)

    ioo, iou = vt.mesh_iou_solid(mesh, obb_part_meshes)
    aioo, aiou = vt.mesh_iou_solid(mesh, aabb_part_meshes)
    # iou, uiou = vt.mesh_iou_sampled(mesh, obb_part_meshes)
    if iou > aiou:
        print("Intersection over Orig Obb {}: {:.3f}".format(i+1, ioo*100))
        visu.vis_mult_seg(lobb_points, lobb_faces, lobb_segs)
    else:
        print("Intersection over Orig AABB {}: {:.3f}".format(i+1, aioo*100))
        visu.vis_mult_seg(laabb_points, laabb_faces, laabb_segs)

    # print("Intersection over Union: {:.3f}".format(iou*100))
    # save point segmentation as extension .pseg
    pseg_fname = "{}.pseg".format(name.split(".")[0])
    ut.save_pseg(save_path, pseg_fname, seg)
    # visu.vis_mult_seg(lst_points, lst_faces, lst_segs)
    