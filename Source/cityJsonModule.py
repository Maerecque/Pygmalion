import open3d as o3d
import sys
import os
# Use alpha shape to find concave boundary (captures inner/underlying edges)


sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from Source.fileHandler import load_and_preprocess_pointcloud
from Source.floorplanFinder import find_boundary_from_floor, sort_points_in_hull, get_keypoints, find_corners
from Source.heightMapModule import transform_pointcloud_to_height_map, create_point_cloud
from Source.linesetTools import contour_to_lineset, filter_lines_within_contour, merge_lineset, lineset_to_trianglemesh
from Source.pointCloudAltering import (
    remove_noise_statistical as rns,
    merge_point_clouds as merge_pcds,
    alter_point_density as apd  # noqa: E501
)
from Source.roofTools import slice_roof_up
from Source.wallTools import (  # noqa: F401
    extract_wall_points,
    keep_wall_points_from_x_height,
    connect_vertically_aligned_points,
    connect_vertically_aligned_points2,
    divide_wall_into_layers
)
from Source.surfaceReconstructor import repair_mesh_with_contour
from Source.pointCloudEditor import open_point_cloud_editor as opce  # noqa: F401

import trimesh
import numpy as np


def repair_mesh(meshes) -> o3d.geometry.TriangleMesh:
    """
    Repairs a non-watertight Open3D TriangleMesh or a list of TriangleMeshes by:
    - combineren (indien lijst) -> één mesh
    - vullen van holes met Trimesh
    - controleren en eventueel omklappen van nieuw aangemaakte faces zodat normals naar buiten wijzen
    Retourneert een gerepareerde Open3D TriangleMesh.
    """
    # allow single mesh or list/tuple of meshes
    if isinstance(meshes, (list, tuple)):
        trimesh_list = []
        for m in meshes:
            trimesh_list.append(
                trimesh.Trimesh(
                    vertices=np.asarray(m.vertices),
                    faces=np.asarray(m.triangles),
                    vertex_colors=(np.asarray(m.vertex_colors) * 255).astype(np.uint8)
                    if m.has_vertex_colors() else None,
                    process=False
                )
            )
        # concatenate into a single Trimesh
        mesh_trimesh = trimesh.util.concatenate(trimesh_list)
    else:
        mesh_o3d = meshes
        mesh_trimesh = trimesh.Trimesh(
            vertices=np.asarray(mesh_o3d.vertices),
            faces=np.asarray(mesh_o3d.triangles),
            vertex_colors=(np.asarray(mesh_o3d.vertex_colors) * 255).astype(np.uint8)
            if mesh_o3d.has_vertex_colors() else None,
            process=False
        )

    # remember original face count to detect newly created faces after fill
    original_face_count = len(mesh_trimesh.faces)

    # 2. Check for and fill any holes
    if not mesh_trimesh.is_watertight:
        print("Holes detected; filling mesh...")
        mesh_trimesh.fill_holes()
    else:
        print("Mesh is already watertight; no action needed.")

    # If new faces were added, ensure their normals point outward
    if len(mesh_trimesh.faces) > original_face_count:
        # clean mesh and compute face normals/centers
        mesh_trimesh.update_faces(mesh_trimesh.unique_faces())
        mesh_trimesh.update_faces(mesh_trimesh.nondegenerate_faces())
        mesh_trimesh.remove_unreferenced_vertices()
        face_normals = mesh_trimesh.face_normals
        face_centers = mesh_trimesh.triangles_center

        # iterate only new faces
        new_indices = np.arange(original_face_count, len(mesh_trimesh.faces))
        for fi in new_indices:
            center = face_centers[fi]
            normal = face_normals[fi]
            # sample a point slightly along the face normal
            sample = center + normal * 1e-3
            # trimesh.contains expects a watertight mesh (we just filled holes),
            # returns True if sample is inside the volume
            try:
                is_inside = mesh_trimesh.contains([sample])[0]
            except Exception:
                # fallback: use ray test - if uncertain, skip flipping
                is_inside = False

            # if the sampled point is inside the mesh, the face normal points inward -> flip face
            if is_inside:
                mesh_trimesh.faces[fi] = mesh_trimesh.faces[fi][::-1]

        # post-process: remove artifacts, re-center and fix normals
        mesh_trimesh.update_faces(mesh_trimesh.unique_faces())
        mesh_trimesh.update_faces(mesh_trimesh.nondegenerate_faces())
        mesh_trimesh.remove_unreferenced_vertices()
        mesh_trimesh.rezero()
        try:
            mesh_trimesh.fix_normals()
        except Exception:
            # if fix_normals isn't available or fails, continue without crashing
            pass

    # 3. Convert the repaired Trimesh object back to Open3D
    repaired_mesh_o3d = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh_trimesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh_trimesh.faces.astype(np.int32))
    )

    # 4. Handle colors and normals
    if hasattr(mesh_trimesh.visual, "vertex_colors") and mesh_trimesh.visual.vertex_colors is not None:
        repaired_mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(
            mesh_trimesh.visual.vertex_colors[:, :3].astype(np.float64) / 255.0
        )

    repaired_mesh_o3d.compute_vertex_normals()

    return repaired_mesh_o3d


def main():
    # Set the verbosity level of Open3D to only print severe errors
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # 1. Load and preprocess the point cloud (user selects file)
    pcd = load_and_preprocess_pointcloud()

    pcd = apd(pcd, points_per_cm=3, print_result=True)

    if pcd is None:
        print("No point cloud loaded. Exiting.")
        return

    # 2. Remove noise from the point cloud
    pcd = rns(pcd, False)

    # 3. Transform the point cloud into a height map (returns tuple: [floor, wall, ...])
    new_pcd_tuple = transform_pointcloud_to_height_map(
        pcd,
        debugging_logs=False
    )

    # 4. Find the boundary lines (hull) of the floor points
    floor_contour = find_boundary_from_floor(new_pcd_tuple[0], 8)

    # 5. Sort the hull points and find corners in the floor boundary
    floor_hull = sort_points_in_hull(floor_contour, 0.045)

    # Find corners
    floor_corners = find_corners(floor_hull, angle_threshold_deg=45, window=2, merge_radius=1)

    # opce(create_point_cloud(floor_corners, color=[1, 0, 0]), show_help=False)

    temp_merge = merge_pcds(
        [
            get_keypoints(create_point_cloud(floor_contour), gamma_21=0.975, gamma_32=0.975, min_neighbors=3, print_stats=False),
            create_point_cloud(floor_corners, color=[1, 0, 0]),
        ]
    )

    full_floor_corners = sort_points_in_hull(temp_merge.points, 0.00005)

    full_floor_corners = floor_hull  # FOR TESTING

    # 6. Create a wall slice at a certain height above the floor
    floor_corners_pcd = create_point_cloud(full_floor_corners, color=[1, 0, 0])  # Red color for corners

    # 7. Extract the roof points above a certain height (removes everything below)
    new_roof_pcd, other_wall_pcd = keep_wall_points_from_x_height(
        new_pcd_tuple[1],
        floor_corners_pcd,
        height=1.5
    )

    temp_wall_pcd = merge_pcds([new_pcd_tuple[2], other_wall_pcd])

    wall_pcd = extract_wall_points(
        temp_wall_pcd,
        floor_corners_pcd,
        search_radius=0.05,
    )

    wall_floor_merge = merge_pcds([floor_corners_pcd, wall_pcd])  # noqa: F841

    # 8. Slice the roof into horizontal slabs and flatten each slice
    sliced_roof_list = slice_roof_up(new_roof_pcd, 30, slab_fatness=0.01, voxel_size=0.05)

    # # This is not used right now but not sure if to delete it
    # floor_mesh_ball_pivoting = reconstruct_mesh_ball_pivoting(floor_corners_pcd, k=30, visualize=True)
    # wall_mesh_ball_pivoting = reconstruct_mesh_ball_pivoting(new_pcd_tuple[2], visualize=True)
    # roof_mesh_ball_pivoting = reconstruct_mesh_ball_pivoting(new_roof_pcd, visualize=True)

    # # Take the first and second list in sliced_roof_list
    # roof_wall_lineset = connect_vertically_aligned_points2(wall_pcd.points, sliced_roof_list, 0.1)

    roof_wall_lineset = o3d.geometry.LineSet()

    wall_layer_list = divide_wall_into_layers(wall_pcd, layer_amount=10)

    opce(merge_pcds(wall_layer_list), show_help=False)

    for i in range(len(wall_layer_list) - 1, 0, -1):
        roof_wall_lineset += connect_vertically_aligned_points2(wall_layer_list[i - 1], wall_layer_list[i], 0.1)
        roof_wall_lineset += contour_to_lineset(sort_points_in_hull(wall_layer_list[i]), max_line_length=0.5)

    # Connect the rest of the roof layers with each other from top to the bottom and per layer create a contour
    for i in range(len(sliced_roof_list) - 1, 0, -1):
        roof_wall_lineset += connect_vertically_aligned_points(sliced_roof_list[i - 1], sliced_roof_list[i], 0.1)
        roof_wall_lineset += contour_to_lineset(sort_points_in_hull(sliced_roof_list[i]), max_line_length=0.5)

    roof_wall_lineset = filter_lines_within_contour(full_floor_corners, roof_wall_lineset)

    # floor_lineset = contour_to_lineset(full_floor_corners)
    # wall_slice_lineset = contour_to_lineset(sort_points_in_hull(wall_pcd.points, 0.00005), 0.5)
    # vertical_lineset = connect_vertically_aligned_points(floor_lineset.points, wall_slice_lineset.points)

    # # Combine vertical and wall slice lineset
    # combined_lineset = merge_lineset(vertical_lineset, wall_slice_lineset)  # noqa: F841

    floor_lineset = contour_to_lineset(full_floor_corners)
    total_lineset = merge_lineset(floor_lineset, roof_wall_lineset)

    part12 = lineset_to_trianglemesh(total_lineset, full_floor_corners)
    part12r = repair_mesh_with_contour(part12, create_point_cloud(full_floor_corners))
    part3 = lineset_to_trianglemesh(floor_lineset, full_floor_corners)

    # Repair meshes
    o3d.visualization.draw(repair_mesh([part12r, part3]))

    # o3d.visualization.draw([
    #     floor_lineset,
    #     combined_lineset,
    #     roof_wall_lineset
    # ])

    o3d.visualization.draw([part12r, part3])

    # export_3d_building_to_cityjson_with_dialog(floor_lineset, combined_lineset, roof_wall_lineset)


if __name__ == "__main__":
    main()
