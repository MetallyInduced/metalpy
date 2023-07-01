<h1 align="center">
<img src="./branding/logo/metalpylogo.png" width="300" alt="METALpy">
</h1>

<div align="center">

[简体中文](README.zh_CN.md) | [English](README.md)

</div>

-------------
![PyPI](https://img.shields.io/pypi/v/metalpy)

**METAL Essential Tools and Libraries (Python)**，简称**metalpy**，
是用于SimPEG以及相关工作流的通用Python工具集，目前包含：
1. MEPA: 一个通用并行化框架
2. [Mexin](metalpy/mexin/README.zh_CN.md): 一个Python注入框架
3. [SCAB](metalpy/scab/README.zh_CN.md): SimPEG相关实用工具和扩展
4. [Carto](metalpy/carto/README.zh_CN.md): 制图相关工具，支持下载瓦片地图与导入导出GeoTIFF图像

安装
------------
metalpy可以通过pip安装：

```console
pip install "metalpy[complete]"
```

<details><summary><b>使用pip安装注意事项</b></summary>
<p>

`metalpy`各个子模块依赖独立管理，因此需要在`pip`安装时指定所需要的模块：

支持的安装参数包括：
```console
pip install "metalpy[complete]"    # 安装全部所需依赖
pip install "metalpy[scab]"        # 安装SCAB所需依赖 
pip install "metalpy[carto]"       # 安装Carto所需依赖
pip install "metalpy[scab, carto]" # 安装SCAB和Carto所需依赖
pip install "metalpy[mepa]"        # 安装MEPA所需依赖
pip install "metalpy[mexin]"       # 安装Mexin所需依赖
pip install "metalpy[tests]"       # 安装测试所需依赖
pip install "metalpy[docs]"        # 安装文档生成所需依赖
pip install "metalpy[dev]"         # 安装开发所需依赖
```

</p>
</details>
