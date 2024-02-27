import pytest
from numpy.testing import assert_equal

from metalpy.scab.modelling import Scene
from metalpy.scab.modelling.shapes import Cuboid


indexing_types = ['index_by_mask', 'index_by_sparse_indices']


def create_mesh(ind_type: str):
    """生成一个长宽高为3个网格的27个网格组成的立方体，
    中心的1个网格被设置为不同的类型，
    构建网格时网格边界向外延申1格（即外部一圈为无效网格）
    """
    scene = Scene()
    cell_size = 1
    cube_cells = 3

    outer_cube = Cuboid(corner=[0, 0, 0], size=cube_cells * cell_size)
    scene.append(outer_cube, models={'value': 1, 'type': 0})
    scene.append(Cuboid(center=outer_cube.center, size=cell_size), models={'value': 2, 'type': 1})
    bounds = scene.bounds

    model_mesh = scene.build(cell_size=cell_size, bounds=bounds.expand(increment=cell_size))
    model_mesh.set_active_model('value')

    if ind_type == indexing_types[1]:
        import numpy as np
        model_mesh.active_cells = np.where(model_mesh.active_mask)[0]

    return model_mesh


@pytest.mark.parametrize('ind_type', indexing_types, indirect=False)
def test_extract(ind_type):
    model_mesh = create_mesh(ind_type=ind_type)
    model_mesh = model_mesh.extract(model_mesh['type'] == 1)
    mask = model_mesh.check_complete_mask([model_mesh.n_cells // 2])  # 即正中间的网格
    assert_equal(model_mesh.active_mask, mask)


@pytest.mark.parametrize('ind_type', indexing_types, indirect=False)
def test_reactivate(ind_type):
    model_mesh = create_mesh(ind_type=ind_type)
    mesh = model_mesh.mesh
    other_model_mesh = model_mesh.reactivate(
        (mesh.cell_centers - mesh.cell_centers[0]).sum(axis=1) <= 6
    )

    import numpy as np
    assert np.count_nonzero(other_model_mesh['value'] == 1) == 16
    assert np.count_nonzero(other_model_mesh['value'] == 2) == 1


@pytest.mark.parametrize('ind_type', indexing_types, indirect=False)
def test_rebind(ind_type):
    model_mesh = create_mesh(ind_type=ind_type)
    other_model_mesh = model_mesh.rebind('type')
    assert other_model_mesh.n_active_cells == 1
