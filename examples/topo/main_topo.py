import tarfile

import numpy as np
import pyvista as pv
from SimPEG import utils
from scipy.interpolate import LinearNDInterpolator

from metalpy.scab.modelling import Scene
from metalpy.scab.modelling.shapes import Terrain, Ellipsoid
from metalpy.utils.bounds import Bounds
from metalpy.utils.file import make_cache_directory_path


def download_data():
    # 例子来自SimPEG稀疏反演教程
    # https://docs.simpeg.xyz/content/tutorials/04-magnetics/plot_inv_2a_magnetics_induced.html

    data_source = "https://storage.googleapis.com/simpeg/doc-assets/magnetics.tar.gz"
    temp_dir = make_cache_directory_path("data")

    # 解压提取地形文件
    downloaded_data = utils.download(data_source, folder=temp_dir)
    tar = tarfile.open(downloaded_data, "r")
    tar.extractall(path=temp_dir)
    tar.close()

    # 载入地形和观测点数据
    topo_xyz = np.loadtxt(temp_dir / "magnetics/magnetics_topo.txt")
    receiver_locations = np.loadtxt(temp_dir / "magnetics/magnetics_data.obs")[:, :3]

    return topo_xyz, receiver_locations


def make_data():
    # 等价定义 （如果下载不了可以改为使用以下代码），来自SimPEG正演教程
    # https://docs.simpeg.xyz/content/tutorials/04-magnetics/plot_2a_magnetics_induced.html

    # 定义地形
    [x_topo, y_topo] = np.meshgrid(np.linspace(-200, 200, 41), np.linspace(-200, 200, 41))
    z_topo = -15 * np.exp(-(x_topo ** 2 + y_topo ** 2) / 80 ** 2)
    x_topo, y_topo, z_topo = utils.mkvc(x_topo), utils.mkvc(y_topo), utils.mkvc(z_topo)
    topo_xyz = np.c_[x_topo, y_topo, z_topo]

    # 定义观测点
    x = np.linspace(-80.0, 80.0, 17)
    y = np.linspace(-80.0, 80.0, 17)
    x, y = np.meshgrid(x, y)
    x, y = utils.mkvc(x.T), utils.mkvc(y.T)
    fun_interp = LinearNDInterpolator(np.c_[x_topo, y_topo], z_topo)
    z = fun_interp(np.c_[x, y]) + 10  # 观测点离地高度10m
    receiver_locations = np.c_[x, y, z]

    return topo_xyz, receiver_locations


if __name__ == '__main__':
    # 构造地形对象并剖分网格
    try:
        topo_xyz, receiver_locations = download_data()
    except OSError:
        topo_xyz, receiver_locations = make_data()

    # 定义目标体
    sphere = Ellipsoid.sphere(15).translated(0, 0, -45)

    # 构建场景
    scene = Scene()
    scene.append(Terrain.xy2z(topo_xyz), models=1e-4)  # 添加地形
    scene.append(sphere, models=0.01)  # 添加目标体

    # 模型所在位置及z方向上采用等距网格
    mesh = scene.create_mesh(cell_size=5.0, bounds=sphere.bounds | Bounds.bounded(zmax=scene.bounds.zmax))
    # 采用指数扩大网格对网格边界进行扩展
    mesh = mesh.expand(scene.bounds, ratio=1.3, increment=Bounds.bounded(zmin=-30))
    mesh = scene.build_model(mesh)

    # 绘制网格
    mesh_grid = mesh.to_polydata()
    pl = pv.Plotter(shape=(1, 2))

    pl.subplot(0, 0)
    pl.add_mesh(mesh_grid, opacity=0.5, show_edges=True)
    pl.add_mesh(mesh_grid.threshold(1e-3))
    pl.add_mesh(receiver_locations)

    pl.subplot(0, 1)
    pl.add_mesh(mesh_grid.slice_orthogonal(x=0, y=0, z=1), show_edges=True)
    pl.add_mesh(receiver_locations)

    pl.link_views()
    pl.show()
