import numpy as np

from metalpy.carto.coords import Coordinates


def min_area_rect_pca(pts, debug=False):
    from sklearn.decomposition import PCA
    from scipy.spatial.transform import Rotation

    pca = PCA(n_components=pts.shape[1])
    pca.fit(pts[::2])

    rotation = np.arctan2(*pca.components_[0, ::-1])
    center = pca.mean_

    proj = Rotation.from_euler('z', -rotation).as_matrix()[:2, :2]

    # (pts - pca.mean_) @ proj.T + pca.mean_  # 将坐标点投影到新坐标系下
    size = Coordinates(pts @ proj.T).bounds.extent  # 尺寸与平移无关，只求边界时可以省略平移

    if debug:
        from matplotlib import pyplot as plt
        plt.scatter(*pts.to_numpy().T, s=0.01)
        plt.scatter(*((pts - pca.mean_) @ proj.T + pca.mean_).to_numpy().T, s=0.01)
        plt.arrow(
            *pca.mean_,
            *pca.components_[0] * 300
        )
        plt.arrow(
            *pca.mean_,
            *pca.components_[1] * 150
        )
        plt.show()

        print(np.prod(Coordinates(pts - center).bounds.extent))
        print(np.prod(Coordinates((pts - pca.mean_) @ proj.T + pca.mean_).bounds.extent))

    return center, size, rotation
