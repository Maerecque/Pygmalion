import numpy as np
import open3d as o3d
import pyvista as pv
import trimesh
from tqdm import tqdm


def compute_distances_to_point_cloud(
    mesh: o3d.cpu.pybind.geometry.TriangleMesh,
    point_cloud: o3d.cpu.pybind.geometry.PointCloud
) -> np.ndarray:
    """Compute the distance from each vertex in the mesh to the nearest point in the point cloud.

    Args:
        mesh (o3d.cpu.pybind.geometry.TriangleMesh): The mesh whose vertices' distances are computed.
        point_cloud (o3d.cpu.pybind.geometry.PointCloud): The original point cloud.

    Returns:
        numpy.ndarray: Array of distances for each vertex in the mesh.
    """
    kd_tree = o3d.geometry.KDTreeFlann(point_cloud)
    distances = []

    for vertex in tqdm(np.asarray(mesh.vertices), desc="Computing distances for down-sampling"):
        _, idx, _ = kd_tree.search_knn_vector_3d(vertex, 1)
        nearest_point = np.asarray(point_cloud.points)[idx[0]]
        distances.append(np.linalg.norm(vertex - nearest_point))

    return np.array(distances)


def filter_vertices_and_faces(
    mesh: o3d.cpu.pybind.geometry.TriangleMesh,
    distances: np.ndarray,
    distance_threshold: float
) -> tuple:
    """Filter out vertices and corresponding faces that are beyond the specified distance.

    Args:
        mesh (o3d.cpu.pybind.geometry.TriangleMesh): The mesh to be filtered.
        distances (numpy.ndarray): Array of distances for each vertex in the mesh.
        distance_threshold (float): Maximum allowed distance for vertices to be kept.

    Returns:
        tuple: Arrays of new vertices, colors, and faces after filtering.
    """
    vertices_to_keep = distances <= distance_threshold
    new_vertices = np.asarray(mesh.vertices)[vertices_to_keep]
    new_colors = np.asarray(mesh.vertex_colors)[vertices_to_keep]
    vertex_map = {i: idx for idx, i in enumerate(np.where(vertices_to_keep)[0])}
    new_faces = [
        [vertex_map[v] for v in face]
        for face in tqdm(mesh.triangles, desc="Filtering faces for down-sampling")
        if all(v in vertex_map for v in face)
    ]

    return new_vertices, new_colors, new_faces


def mesh_simple_downsample(
    mesh: o3d.cpu.pybind.geometry.TriangleMesh,
    original_point_cloud: o3d.cpu.pybind.geometry.PointCloud,
    distance_threshold: float = 0.05,
    visualize_mesh: bool = False,
) -> o3d.cpu.pybind.geometry.TriangleMesh:
    """Simplifies a mesh by removing vertices that are far from the original point cloud.

    Args:
        mesh (o3d.cpu.pybind.geometry.TriangleMesh): The mesh to be simplified.
        original_point_cloud (o3d.cpu.pybind.geometry.PointCloud): The original point cloud.
        distance_threshold (float, optional): Maximum distance between a vertex in the mesh and
            the nearest point in the point cloud. Defaults to 0.05.
        visualize_mesh (bool, optional): Whether to visualize the simplified mesh. Defaults to False.

    Raises:
        ValueError: If the mesh or the point cloud is empty.

    Returns:
        o3d.cpu.pybind.geometry.TriangleMesh: The simplified mesh.
    """
    if not mesh:
        raise ValueError("The mesh is empty.")

    if not original_point_cloud:
        raise ValueError("The point cloud is empty.")

    print("Starting mesh simplification...")

    # Step 1: Compute distances from mesh vertices to the point cloud
    distances = compute_distances_to_point_cloud(mesh, original_point_cloud)

    # Step 2: Filter vertices and faces based on distance threshold
    new_vertices, new_colors, new_faces = filter_vertices_and_faces(mesh, distances, distance_threshold)

    # Step 3: Create a new mesh with filtered vertices and faces
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    new_mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)
    new_mesh.triangles = o3d.utility.Vector3iVector(new_faces)

    # Optional: Recompute normals for the new mesh
    new_mesh.compute_vertex_normals()
    new_mesh.compute_triangle_normals()

    # Smoothing the mesh
    smooth_mesh = new_mesh.filter_smooth_simple(number_of_iterations=15)

    if visualize_mesh:
        o3d.visualization.draw_geometries([smooth_mesh], mesh_show_back_face=True)

    print(f"Removed {len(mesh.triangles) - len(smooth_mesh.triangles)} triangles")

    return smooth_mesh


def transform_pcd_to_mesh(
    input_pcd: o3d.cpu.pybind.geometry.PointCloud,
    alpha: float = 0.1,
    tolerance: float = 0.05,
    offset: float = 0.0,
    visualize_bool: bool = False,
    bool_3d_mesh: bool = True
) -> pv.UnstructuredGrid:
    """Transforms a point cloud into a mesh using the Delaunay algorithm.

    Args:
        input_pcd (o3d.cpu.pybind.geometry.PointCloud): The point cloud to be transformed into a mesh.
        alpha (float, optional): Alpha value for the Delaunay algorithm. Defaults to 0.1.
        tolerance (float, optional): Tolerance value for the Delaunay algorithm. Defaults to 0.05.
        offset (float, optional): Offset value for the Delaunay algorithm. Defaults to 0.0.
        visualize_bool (bool, optional): Whether to visualize the mesh. Defaults to False.
        bool_3d_mesh (bool, optional): Whether to create a 3D mesh (True) or a 2D mesh (False). Defaults to True.

    Returns:
        pv.UnstructuredGrid: The mesh created from the point cloud.
    """
    # Convert the point cloud to a numpy array
    whole_cloud_points = np.asarray(input_pcd.points)

    # Create a PyVista PolyData object from the point cloud
    cloud = pv.PolyData(whole_cloud_points)

    # Apply the Delaunay algorithm to create a mesh
    if bool_3d_mesh:
        volume = cloud.delaunay_3d(alpha=alpha, progress_bar=True, tol=tolerance, offset=offset)
    else:
        volume = cloud.delaunay_2d(alpha=alpha, progress_bar=True, tol=tolerance, offset=offset)

    # Visualize the resulting mesh if requested
    if visualize_bool:
        shell = volume.extract_geometry(progress_bar=True)
        shell.plot()

    return volume


def repair_mesh(meshes) -> o3d.geometry.TriangleMesh:
    """Repair holes in a mesh or list of meshes and ensure face normals point outward.

    Args:
        meshes (o3d.geometry.TriangleMesh or list/tuple of o3d.geometry.TriangleMesh): The mesh or meshes to repair.

    Returns:
        o3d.geometry.TriangleMesh: The repaired mesh with holes filled and normals corrected.

    Usage:
        mesh = o3d.io.read_triangle_mesh("path_to_mesh.ply")
        repaired_mesh = repair_mesh(mesh)
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


def o3d_to_cityjson(
    mesh: o3d.geometry.TriangleMesh,
    cityobject_id: str = "obj1",
    obj_type: str = "Building",
    lod: str = "1.0",
) -> dict:
    """Convert an Open3D TriangleMesh into a minimal CityJSON object.

    This function extracts the vertices and triangle faces from an
    Open3D TriangleMesh and reformats them into a CityJSON-compliant
    dictionary. The geometry is wrapped as a Solid with triangular
    boundaries.

    Args:
        mesh (o3d.geometry.TriangleMesh): The input Open3D mesh.
        cityobject_id (str, optional): Identifier for the CityObject.
            Defaults to "obj1".
        obj_type (str, optional): The CityJSON object type (e.g.,
            "Building", "TINRelief"). Defaults to "Building".
        lod (str, optional): Level of detail of the geometry.
            Defaults to "1.0".

    Returns:
        dict: A CityJSON object containing vertices and geometry
        definitions.
    """
    vertices = np.asarray(mesh.vertices).tolist()
    faces = np.asarray(mesh.triangles).tolist()

    cityjson = {
        "type": "CityJSON",
        "version": "1.1",
        "CityObjects": {
            cityobject_id: {
                "type": obj_type,
                "geometry": [
                    {
                        "type": "Solid",
                        "lod": lod,
                        "boundaries": [[[face] for face in faces]],
                    }
                ],
            }
        },
        "vertices": vertices,
    }
    return cityjson
