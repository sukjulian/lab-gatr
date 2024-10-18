from setuptools import find_packages, setup


minimal_installation_requirements = [
    "torch",
    "xformers",
    "torch_geometric",
    "torch_scatter",
    "torch_cluster",
    "gatr @ git+https://github.com/Qualcomm-AI-research/geometric-algebra-transformer.git"
]

setup(
    name="lab_gatr",
    version="0.4.2",
    author="Julian Suk",
    author_email="j.m.suk@utwente.nl",
    license="MIT",
    packages=find_packages(),
    install_requires=minimal_installation_requirements
)
