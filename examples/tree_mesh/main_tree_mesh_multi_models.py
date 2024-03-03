import numpy as np
import pyvista

from metalpy.scab.modelling import Scene, MixMode
from metalpy.scab.modelling.shapes import Obj2, Ellipsoid
from metalpy.scab.modelling.shapes.obj2 import OriginScheme
from metalpy.utils.time import timed


def main():
    bunny = pyvista.examples.downloads.download_bunny()
    bunny.flip_normals()
    scene = Scene.of(
        Obj2(
            bunny,
            size=[15.6, 15.4, 12.1],
            surface_range=[None, 0],
            subdivide=True,
            reset_origin=OriginScheme.Center
        ),
        models={'homogeneous': 1, 'inhomogeneous': 1}
    )
    scene.append(
        Ellipsoid(*[10, 10, 10]),
        models={'homogeneous': 1, 'inhomogeneous': 2},
        mix_mode=MixMode.KeepOriginal
    )

    n_cells = np.asarray([32, 31, 24]) * 2

    with timed():
        refined_mesh = scene.build_tree(n_cells=n_cells, method=Scene.Octree.refine)
        print(f'Active cells of refined mesh: {refined_mesh.n_active_cells}')

    with timed():
        simplified_mesh = scene.build_tree(n_cells=n_cells, method=Scene.Octree.simplify)
        print(f'Active cells of simplified mesh: {simplified_mesh.n_active_cells}')

    pl = pyvista.Plotter(shape=(2, 2))

    pl.subplot(0, 0)
    pl.add_title('Refine (homogeneous)')
    pl.add_mesh(scene.to_multiblock(), color='white', opacity=0.3)
    pl.add_mesh(refined_mesh.to_polydata(scalars='homogeneous', prune=False).slice_orthogonal(), show_edges=True)

    pl.subplot(0, 1)
    pl.add_title('Simplify (homogeneous)')
    pl.add_mesh(scene.to_multiblock(), color='white', opacity=0.3)
    pl.add_mesh(simplified_mesh.to_polydata(scalars='homogeneous', prune=False).slice_orthogonal(), show_edges=True)

    pl.subplot(1, 0)
    pl.add_title('Refine (inhomogeneous)')
    pl.add_mesh(scene.to_multiblock(), color='white', opacity=0.3)
    pl.add_mesh(refined_mesh.to_polydata(scalars='inhomogeneous', prune=False).slice_orthogonal(), show_edges=True)

    pl.subplot(1, 1)
    pl.add_title('Simplify (inhomogeneous)')
    pl.add_mesh(scene.to_multiblock(), color='white', opacity=0.3)
    pl.add_mesh(simplified_mesh.to_polydata(scalars='inhomogeneous', prune=False).slice_orthogonal(), show_edges=True)

    pl.link_views()
    pl.show()


if __name__ == '__main__':
    main()
