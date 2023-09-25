# Artist vs The Machine: can you beat an AI at Pictionary?

This repository contains materials needed to run an interactive tutorial on utilizing Artificial Intelligence (AI) for atmospheric science titled "Artist vs. The Machine: can you beat an AI at Pictionary?".
Briefly, the tutorial demonstrates the usefulness of image classification for scientific applications by creating and demoing a model - named DAISY - that can identify simply sketches of weather patterns, then walks students through some guided discussion questions that make the connection between sketch classification (i.e. Pictionary) and real-world applications in meteorology.
The tutorial is aimed at middle to high school level students, and contains an interactive component where students create their own sketches online and the instructor inputs these sketches into the AI and shows the results in real-time.
It uses Google's [Quick Draw!](https://quickdraw.withgoogle.com) Dataset for training the AI model and was originally developed by scientists at the Global Systems Laboratory of the National Oceanic at Atmospheric Administration (NOAA) and the Cooperative Institute for Research in Earth Sciences (CIRES) at the University of Colorado Boulder.

The repository contains all of the code necessary for downloading the Quick Draw data, preprocessing the Quick Draw data, preprocessing student sketches, building and running the model, running some basic diagnostics, and displaying the sketches in various forms.
It also contains a jupyter notebook that can be run during the tutorial, and a set of accompanying slides, as well as a PDF guide for instructors on how to use the various materials in a tutorial and the materials required.

Although untested, the tutorial could be easily modified for other scientific applications by modifying the appropriate content in the accompanying presentation, and modifying the [categories](categories.txt) we pull the training data from.
For example, an oceanic science application might use drawings from the canoe, cruise ship, dolphin, sailboat, shark, and whale categories, and discuss satellite image identification of types of sea creatures vs. boats and commercial vs. leisure boats, etc.
A biology application might use drawings from the tiger, lion, dog, crocodile, frog and hedgehog categories, and discuss automatic identification of species from trail cameras.

Please note that the raw data set is pulled directly from Google and while it was individually moderated, it may still contain inappropriate content.
Any images that are created with the weather categories and using all the pre-set random seeds, numbers of samples, and indices in this version of the code, have been checked for inappropriate content.
Before giving this tutorial, it is advised that you check all the generated images in the slides and tutorial notebook.

A hosted version of the tutorial is available on Binder.
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kirana-bergstrom/artist-vs-machine/HEAD?labpath=notebook_tutorial.ipynb)

## Content
- [Getting started](#getting-started)
- [Pulling the raw data](#pulling-the-raw-data)
- [Running the code](#running-the-code)
- [Tutorial notebook and slides](#tutorial-notebook-and-slides)
- [Instructor tutorial guide](#instructor-tutorial-guide)
- [Contributing](#contributing)

## Getting started
Pull the repo and navigate to it using
```sh
$ git pull https://github.com/artist-vs-the-machine
$ cd artist-vs-the-machine
```
Create and activate a conda environment with the necessary requirements using the commands
```sh
$ conda create artist_v_machine
$ conda create artist_v_machine
```

In the top level of the repository, the most important file is the "categories.txt" file.
It can be modified to include a different subset of categories from the Quick, Draw! dataset categories as desired.

After setting up the environment, run
```sh
$ ./setup.sh
```
to download the raw data and create the necessary directory structure.
Downloading the raw data takes some time, and should be done before actually running the tutorial.

## Running the code
Code for building, running the model and some diagnostics is contained in run.py.
Certain pieces of the code can be run during the tutorial, but the preprocessing step takes a significant amount of time and is recommended to be run before the tutorial so the data is ready.
```sh
$ python run.py no_preprocess no_build
```
Building the model during the tutorial is more instructive, but also takes some time. It can be done before the tutorial as well.

## Tutorial notebook and slides
Run the jupyter notebook using the command
```sh
$ jupyter notebook ./tutorial/atmospheric_tutorial.ipynb
```
Create the accompanying slideshow using the command

and open it in an html viewer.

## Instructor tutorial guide
A PDF is included in this repository for instructors to use here.
A sample set of simple instructions to give to students for the interactive portion of the tutorial is located here.

## Contributing
Contributions are welcome.
Bug fixes and code improvements can be submitted via pull request.
To add additional content including modified presentations for other applications, please email [kirana.bergstrom@noaa.gov](mailto:kirana.bergstrom@noaa.gov) before submitting a pull request.
