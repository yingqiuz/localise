# localise

[![Build Status](https://git.fmrib.ox.ac.uk/yqzheng1/python-localise/badges/main/pipeline.svg)](https://git.fmrib.ox.ac.uk/yqzheng1/python-localise)
[![Codecov](https://git.fmrib.ox.ac.uk/yqzheng1/python-localise/badges/main/coverage.svg)](https://git.fmrib.ox.ac.uk/yqzheng1/python-localise)
[![Documentation Status](https://readthedocs.org/projects/localise/badge/?version=latest)](https://localise.readthedocs.io/en/latest/?badge=latest)

This library implements LOCALISE, a python toolbox developed to address the challenges associated with accurately targeting DBS targets on low-quality clinical-like dataset. This toolbox uses Image Quality Transfer techniques to transfer anatomical information from high-quality data to a wide range of connectivity features in low-quality data. The goal is to augment the inference on DBS targets localisation even with compromised data quality. We also have a Julia implementation [here](https://git.fmrib.ox.ac.uk/yqzheng1/hqaugmentation.jl). For more details, please check our paper [here](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_17).
