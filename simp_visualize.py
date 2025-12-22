import os
import argparse
import util as ut
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
save_path = os.path.join(data_path, args.save_dir)

gt_path = ""
seg_res_path = ""
gt_path = os.path.join(data_path, "gt_conn") # "gt_simp"
seg_res_path = os.path.join(data_path, "seg_conn") # "seg_simp"

# make edge saving directory
os.makedirs(save_path, exist_ok=True)

meshes, names = ut.read_mesh(gt_path) # read all meshes / sorted
point_seg, _, _ = ut.read_seg_res(seg_res_path)

for i in range(379,380): # len(meshes)
    mesh = meshes[i]
    name = names[i]
    points = ut.get_vertex(mesh)
    faces = ut.get_face(mesh)
    cur_seg = point_seg[i] # get related segmentation

    # Princeton Seg Dataset has several label per one mesh
    # There are two options that you can view the label
    # 1: Use this code to view specific label per mesh
    # num_lst = [12, 10, 2, 0, 4, 4, 0, 4, 9, 5, 4, 7, 0, 4, 2, 1, 7, 9, 3, 10] # class 1 new
    # elem = num_lst[i%20]
    # seg = cur_seg[elem]
    # visu.vis_face_seg(points, faces, seg)
    # print("Seg Name: {}_{}.seg, Seg Num: {}".format(name.split(".")[0], elem, len(set(seg))))

    # 2: Uncomment this to view all mesh-label pair when segmentation number < 45
    for j in range(len(cur_seg)):
        seg = cur_seg[j]
        if seg.max() < 45:
            visu.vis_face_seg(points, faces, seg)
            print("Seg Name: {}_{}.seg, Seg Num: {}".format(name.split(".")[0], j, len(set(seg))))
        