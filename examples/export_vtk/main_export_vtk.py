import pyvista as pv
from scipy.spatial.transform import Rotation

from metalpy.scab.modelling import Scene
from metalpy.scab.modelling.shapes import Cuboid, Prism, Ellipsoid, Tunnel, Obj2, BarFramework
from metalpy.utils.bounds import bounded


def main():
    scene = Scene.of(
        BarFramework(Cuboid([1, 1, 1], size=2), 0.2, n_rooms=2),
        Prism([[0, 0], [-2, 0], [-2, 1], [-1, 1]], 1, 3),
        Prism([[0, 0], [-2, 0], [-2, 1], [-1, 1]], 1, 3).rotated(90, 0, 0, degrees=True),
        Ellipsoid.spheroid(1, 3, 0).translated(0, -2, 2),
        Tunnel([-3, 2, 2], 0.5, 1, 3),
        Obj2('./stl_models/mine.stl', scale=0.03, surface_thickness=0.4, subdivide=True).translated(1, -2, -1),
        models=1
    ).with_background(1e-5)

    # 使用bounded裁剪场景
    model_mesh = scene.build(cell_size=0.1, cache=True, bounds=bounded(zmax=2, zmin=0))
    grids = model_mesh.to_polydata()

    p = pv.Plotter(shape=(1, 3))

    models = scene.to_multiblock()
    active_grids = grids.threshold([1, 1])

    p.subplot(0, 0)
    p.add_mesh(models, color='white')
    p.show_grid()
    p.show_axes()

    c = p.camera.position

    p.subplot(0, 1)
    p.add_mesh(active_grids, show_edges=True)
    p.show_grid()
    p.show_axes()

    p.subplot(0, 2)
    p.add_mesh(models, opacity=0.4, color='white')
    p.add_mesh(active_grids, show_edges=True)
    p.add_mesh(grids.threshold([0, 1]), show_edges=True, opacity=0.2)
    p.show_grid()
    p.show_axes()

    p.link_views()
    p.renderers[0].camera.position = c
    p.show()


if __name__ == '__main__':
    main()
