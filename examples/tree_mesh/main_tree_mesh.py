import numpy as np
import pyvista

from metalpy.scab.modelling import Scene
from metalpy.scab.modelling.shapes import Obj2
from metalpy.utils.time import timed


def main():
    bunny = pyvista.examples.downloads.download_bunny()
    bunny.flip_normals()
    scene = Scene.of(Obj2(bunny, size=[15.6, 15.4, 12.1], surface_range=[None, 0], subdivide=True))

    n_cells = np.asarray([32, 31, 24]) * 2

    with timed():
        refined_mesh = scene.build_tree(n_cells=n_cells, method=Scene.Octree.refine)
        print(f'Active cells of refined mesh: {refined_mesh.n_active_cells}')

    with timed():
        simplified_mesh = scene.build_tree(n_cells=n_cells, method=Scene.Octree.simplify)
        print(f'Active cells of simplified mesh: {simplified_mesh.n_active_cells}')

    pl = pyvista.Plotter(shape=(1, 2))

    pl.subplot(0, 0)
    pl.add_title('Refine')
    pl.add_mesh(scene.to_multiblock(), opacity=0.7)
    pl.add_mesh(refined_mesh.to_polydata(prune=False).slice_orthogonal(), show_edges=True)

    pl.subplot(0, 1)
    pl.add_title('Simplify')
    pl.add_mesh(scene.to_multiblock(), opacity=0.7)
    pl.add_mesh(simplified_mesh.to_polydata(prune=False).slice_orthogonal(), show_edges=True)

    pl.link_views()
    pl.show()


if __name__ == '__main__':
    main()
