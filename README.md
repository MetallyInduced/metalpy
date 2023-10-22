<h1 align="center">
<img src="./branding/logo/metalpylogo.png" width="300" alt="METALpy">
</h1>

<div align="center">

[简体中文](README.zh_CN.md) | [English](README.md)

</div>

-------------
![PyPI](https://img.shields.io/pypi/v/metalpy)

**METAL Essential Tools and Libraries (Python)** (**metalpy**),
is a collection of common python tools and libraries for SimPEG and related workflows,
which currently includes:
1. [**MEPA**](metalpy/mepa/README.md): a general-purposed parallelization framework.
2. [**Mexin**](metalpy/mexin/README.md): a code injection framework for Python.
3. [**SCAB**](metalpy/scab/README.md): a collection of SimPEG related utilities and extensions.
4. [**Carto**](metalpy/carto/README.md): cartography related utilities, aiming to download tile maps and
save/load GeoTIFF images.

Installation
------------
Metalpy can be installed using _pip_:

```console
pip install "metalpy[complete]"
```

<details><summary><b>Notes on installing with pip</b></summary>
<p>

`metalpy` manages its dependencies separately, which means expected modules 
should be specified in `pip` installation process.

Supported commands are listed next:
```console
pip install "metalpy[complete]"    # Install all requirements
pip install "metalpy[scab]"        # Install requirements for SCAB
pip install "metalpy[carto]"       # Install requirements for Carto
pip install "metalpy[scab, carto]" # Install requirements for SCAB and Carto
pip install "metalpy[mepa]"        # Install requirements for MEPA
pip install "metalpy[mexin]"       # Install requirements for Mexin
pip install "metalpy[tests]"       # Install requirements for tests
pip install "metalpy[docs]"        # Install requirements for doc generation
pip install "metalpy[dev]"         # Install requirements for development
```

</p>
</details>
