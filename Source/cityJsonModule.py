import open3d as o3d
import sys
import os
# Use alpha shape to find concave boundary (captures inner/underlying edges)


sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from Source.fileHandler import load_and_preprocess_pointcloud
from Source.floorplanFinder import find_boundary_from_floor, sort_points_in_hull
from Source.heightMapModule import transform_pointcloud_to_height_map, create_point_cloud
from Source.linesetTools import contour_to_lineset, filter_lines_within_contour, merge_lineset, lineset_to_trianglemesh
from Source.meshAlterer import o3d_to_cityjson, repair_mesh
from Source.pointCloudAltering import (
    remove_noise_statistical as rns,
    merge_point_clouds as merge_pcds,
    alter_point_density as apd
)
from Source.roofTools import slice_roof_up
from Source.wallTools import (
    extract_wall_points,
    keep_wall_points_from_x_height,
    connect_vertically_aligned_points,
    connect_vertically_aligned_points2,
    divide_wall_into_layers
)
from Source.surfaceReconstructor import repair_mesh_with_contour
from Source.pointCloudEditor import open_point_cloud_editor as opce  # Keep here  # noqa: F401

from tqdm import tqdm


def main():
    # Set the verbosity level of Open3D to only print severe errors
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # 1. Load and preprocess the point cloud (user selects file)
    pcd = load_and_preprocess_pointcloud()

    pcd = apd(pcd, points_per_cm=1, print_result=True)

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
    full_floor_contour = find_boundary_from_floor(new_pcd_tuple[0], 8)

    # 5. Sort the hull points and find corners in the floor boundary
    full_floor_corners = sort_points_in_hull(full_floor_contour, 0.045)

    # 6. Create a point cloud of the floor corners for visualization and further processing
    full_floor_corners_pcd = create_point_cloud(full_floor_corners, color=[1, 0, 0])  # Red color for corners

    # 7. Extract the roof points above a certain height (removes everything below)
    new_roof_pcd, other_wall_pcd = keep_wall_points_from_x_height(
        new_pcd_tuple[1],
        full_floor_corners_pcd,
        height=1.5
    )

    temp_wall_pcd = merge_pcds([new_pcd_tuple[2], other_wall_pcd])

    wall_pcd = extract_wall_points(
        temp_wall_pcd,
        full_floor_corners_pcd,
        search_radius=0.05,
    )

    # Empty temporary wall point cloud to save memory
    temp_wall_pcd = None

    opce(merge_pcds([full_floor_corners_pcd, wall_pcd, new_roof_pcd]), show_help=False)

    roof_wall_lineset = o3d.geometry.LineSet()

    wall_layer_list = divide_wall_into_layers(wall_pcd, layer_amount=20)

    # Connect the wall layers with each other from bottom to top and per layer create a contour
    for i in tqdm(range(len(wall_layer_list)), desc="Processing wall layers", unit="layer"):
        roof_wall_lineset += connect_vertically_aligned_points2(
            wall_layer_list[i - 1] if i > 0 else wall_layer_list[i], wall_layer_list[i], 0.1)
        roof_wall_lineset += contour_to_lineset(sort_points_in_hull(wall_layer_list[i]), max_line_length=0.5)

    # 8. Slice the roof into horizontal slabs and flatten each slice
    sliced_roof_list = slice_roof_up(new_roof_pcd, 50, slab_fatness=0.01, voxel_size=0.05)

    # Connect the rest of the roof layers with each other from top to the bottom and per layer create a contour
    for i in range(len(sliced_roof_list) - 1, 0, -1):
        roof_wall_lineset += connect_vertically_aligned_points(sliced_roof_list[i - 1], sliced_roof_list[i], 0.1)
        roof_wall_lineset += contour_to_lineset(sort_points_in_hull(sliced_roof_list[i]), max_line_length=0.5)

    roof_wall_lineset = filter_lines_within_contour(full_floor_corners, roof_wall_lineset)

    floor_lineset = contour_to_lineset(full_floor_corners)
    total_lineset = merge_lineset(floor_lineset, roof_wall_lineset)

    part12 = lineset_to_trianglemesh(total_lineset, full_floor_corners)

    part12r = repair_mesh_with_contour(part12, create_point_cloud(full_floor_corners))
    part3 = lineset_to_trianglemesh(floor_lineset, full_floor_corners)

    repaired = repair_mesh([part12r, part3])

    o3d.visualization.draw([part12r, part3])

    cityjson_data = o3d_to_cityjson(repaired, cityobject_id="building_1", obj_type="Building", lod="1.0")

    # For testing: save to file
    import json
    with open("output_building.json", "w") as f:
        json.dump(cityjson_data, f, indent=2)

    print("CityJSON data created and saved to output_building.json")

    # export_3d_building_to_cityjson_with_dialog(floor_lineset, combined_lineset, roof_wall_lineset)


if __name__ == "__main__":
    main()
