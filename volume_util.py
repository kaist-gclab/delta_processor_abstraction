import numpy as np
import trimesh
import open3d as o3d
from trimesh import repair


def split_by_face_label(mesh, labels):
    """_summary_

    Args:
        mesh (trimesh): contains points, faces, vertex, normal ...
        labels (ndarray): label for mesh faces

    Returns:
        parts: dictionary of submesh (parts)
    """
    parts = {}
    for clabel in np.unique(labels):
        clabel = int(clabel)
        mask = (labels == clabel)
        face_idx = np.nonzero(mask)[0]

        part_mesh = mesh.submesh([face_idx], append=True)
        parts[clabel] = part_mesh

    return parts


def fill_hole(part_mesh):
    """_summary_: fills hole for part mesh

    Args:
        part_mesh (_type_): _description_

    Returns:
        _type_: _description_
    """
    repair.fill_holes(part_mesh)             # caps boundary loops
    repair.fix_normals(part_mesh)            # consistent orientation
    if hasattr(part_mesh, "remove_degenerate_faces"):
        part_mesh.remove_degenerate_faces()
    else:
        # older trimesh: use nondegenerate_faces mask/indices
        if hasattr(part_mesh, "nondegenerate_faces"):
            nd = part_mesh.nondegenerate_faces
            # some versions make this a property, some a method
            if callable(nd):
                nd = nd()
            # nd can be mask or indices; update_faces accepts both
            part_mesh.update_faces(nd)
            part_mesh.remove_unreferenced_vertices()
            
    if hasattr(part_mesh, "remove_duplicate_faces"):
        part_mesh.remove_duplicate_faces()
    else:
        pass

    mesh_vol = abs(part_mesh.volume)

    return part_mesh, mesh_vol


def calculate_obb(part_mesh):
    """_summary_

    Args:
        part_mesh (trimesh object): part of the mesh
    Returns:
        obb (trimesh Box object): 
        obb_vol (float): bounding box volume
    """
    obb = part_mesh.bounding_box_oriented # trimesh.primitives.Box
    obb_extents = obb.extents # length of side (3,)
    obb_vol = obb_extents.prod()

    return obb, obb_vol


# Outlier은 제거하고 bounding box를 계산
def obb_without_outliers(points: np.ndarray, outlier_ratio: float = 0.05):
    """
    Args:
        points: (N, 3) numpy array
        outlier_ratio: fraction of farthest points to drop (e.g. 0.05 for 5%)
    Returns:
        obb (open3d): oriented bounding box
        obb_vol (float): bounding box volume
    """
    pts = np.asarray(points)
    assert pts.ndim == 2 and pts.shape[1] == 3

    if len(pts) < 4:
        # too few points, just use all
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        obb = pcd.get_oriented_bounding_box()
        obb_vol = float(np.prod(obb.extent))
        return obb, obb_vol

    # 1) center (mean) – you can use median if you want more robustness
    center = pts.mean(axis=0)

    # 2) distance from center
    dists = np.linalg.norm(pts - center, axis=1)

    # 3) keep the closest (1 - outlier_ratio) points
    cutoff = np.quantile(dists, 1.0 - outlier_ratio)  # 95% quantile
    inliers = pts[dists <= cutoff]

    if len(inliers) < 4:
        # fallback if we removed too many
        inliers = pts

    # 4) build OBB from inliers
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inliers))
    obb = pcd.get_oriented_bounding_box()
    obb_vol = float(np.prod(obb.extent))

    return obb, obb_vol


def aabb_without_outliers(points: np.ndarray, outlier_ratio: float = 0.05):
    """
    Args:
        points: (N, 3) numpy array
        outlier_ratio: fraction of farthest points to drop (e.g. 0.05 for 5%)

    Returns:
        aabb (open3d.geometry.AxisAlignedBoundingBox)
        aabb_vol (float): bounding box volume
    """
    pts = np.asarray(points)
    assert pts.ndim == 2 and pts.shape[1] == 3

    if len(pts) < 10:
        # too few points, just use all
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        aabb = pcd.get_axis_aligned_bounding_box()
        extents = np.asarray(aabb.get_extent())
        aabb_vol = float(np.prod(extents))
        return aabb, aabb_vol

    # 1) center (mean) – same as your OBB version
    center = pts.mean(axis=0)

    # 2) distance from center
    dists = np.linalg.norm(pts - center, axis=1)

    # 3) keep the closest (1 - outlier_ratio) points
    cutoff = np.quantile(dists, 1.0 - outlier_ratio)
    inliers = pts[dists <= cutoff]

    if len(inliers) < 3:
        # fallback if we removed too many
        inliers = pts

    # 4) build AABB from inliers
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inliers))
    aabb = pcd.get_axis_aligned_bounding_box()
    extents = np.asarray(aabb.get_extent())
    aabb_vol = float(np.prod(extents))

    return aabb, aabb_vol


# calculate bounding box and change size by ratio
def inner_obb_from_obb(obb: trimesh.primitives.Box, ratioh=1.0, ratio=0.8):
    # 1) get oriented side lengths (in box local frame)
    ext = obb.primitive.extents  # (3,) float

    scale = np.ones(3)
    scale[0] *= ratio
    scale[1] *= ratio
    scale[2] *= ratioh

    # 2) shrink them
    inner_extents = scale * ext

    # 3) use the same transform (same center + orientation)
    T = obb.primitive.transform

    # 4) create a new Box primitive
    inner_obb = trimesh.primitives.Box(extents=inner_extents, transform=T)

    inner_vol = float(np.prod(inner_extents))

    return inner_obb, inner_vol



def open3d_obb_to_trimesh_box(obb) -> trimesh.primitives.Box:
    """_summary_: Convert an Open3D OrientedBoundingBox to a trimesh Box mesh.

    Args:
        obb (open3d.geometry.Geometry): o3d box mesh

    Returns:
        box( trimesh.primitives.Box): trimesh box mesh
    """
    center = np.asarray(obb.center)   # or np.asarray(obb.get_center())
    extents = np.asarray(obb.extent)  # (dx, dy, dz)
    R = np.asarray(obb.R)             # 3x3 rotation

    # Build 4x4 transform
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center

    # Create primitive box in that pose
    box = trimesh.primitives.Box(extents=extents, transform=T)

    return box


def open3d_aabb_to_trimesh_box(aabb) -> trimesh.primitives.Box:
    """_summary_

    Args:
        aabb (_type_): _description_

    Returns:
        trimesh.primitives.Box: _description_
    """
    # Open3D AABB API
    center  = np.asarray(aabb.get_center())   # (3,)
    extents = np.asarray(aabb.get_extent())   # (dx, dy, dz)

    # Axis-aligned → rotation = identity
    R = np.eye(3)

    # Build 4x4 transform
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center

    # Trimesh primitive box at that pose
    box = trimesh.primitives.Box(extents=extents, transform=T)
    return box


def mesh_iou_solid(orig_mesh, obb_part_list, engine="blender"):
    # boxes = [open3d_obb_to_trimesh_box(obb) for obb in obb_part_list] # open3d box meshes
    # obb_mesh = trimesh.boolean.union(boxes, engine=engine) # gathered box meshes
    obb_mesh = trimesh.boolean.union(obb_part_list, engine=engine) # gathered box meshes
    # print(orig_mesh.is_watertight)
    # print(obb_mesh.is_watertight)

    mesh_union = trimesh.boolean.union([orig_mesh, obb_mesh], engine=engine)
    mesh_inter = trimesh.boolean.intersection([orig_mesh, obb_mesh], engine=engine, check_volume=True)
    # print(mesh_union.is_watertight)
    # print(mesh_inter.is_watertight)
    
    if mesh_inter is None:
        return 0.0
    # elif isinstance(mesh_inter, trimesh.Trimesh):
    #     mesh_inter = [mesh_inter]
    # else:
    #     mesh_inter = mesh_inter
    
    v_orig = orig_mesh.volume
    v_union = mesh_union.volume
    v_inter = mesh_inter.volume
    # v_inter = sum(m.volume for m in mesh_inter)


    if v_orig <= 0:
        return 0.0
    
    return float(v_inter/v_orig), float(v_inter/v_union)