from metalpy.scab.modelling import ModelledMesh
from metalpy.scab.modelling.shapes import Ellipsoid
from metalpy.utils.file import make_cache_file


def main():
    tmp_filepath = make_cache_file('temp.meshz')

    mesh = Ellipsoid(5, 6, 7).build(1, cell_size=1)
    mesh.to_meshz(tmp_filepath)

    mesh2 = ModelledMesh.from_meshz(tmp_filepath)
    mesh2.plot(show_edges=True)


if __name__ == '__main__':
    main()
