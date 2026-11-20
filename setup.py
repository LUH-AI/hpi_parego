from setuptools import setup

setup(
    name="pckg_hpi_parego",
    packages=["hpi_parego"],
    author="anonymous",
    install_requires=[
        "numpy>=1.26.4",
        "scipy>=1.14.0",
        "smac~=2.3.1",
        "pandas~=2.2.2",
        "configspace~=1.2.1",
        "hypershap~=0.0.8",
        "pymoo~=0.6.1",
    ],
    extras_require={
        "qnparego": ["botorch~=0.16.1"],
    },
)
