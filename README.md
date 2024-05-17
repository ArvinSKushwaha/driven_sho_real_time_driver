# Driven SHO Real Time Driver

This project aims to make an accurate physical model of an agent coupled to the
boundary of a two-dimensional grid of harmonic oscillators.

This repository contains two parts (for now):

- `simulator`: A C++ project for simulating a 2-dimensional grid of harmonic
oscillators with coupling to some driving forces on the boundary of the grid.
Python binding are also provided.

- `driving_agent`: A Python project using
[PyTorch](https://github.com/pytorch/pytorch) to train a model for the value
function of the system without using information on the harmonic oscillator
system itself.

These two parts are rather disjoint for now. For instructions on how to build
each section, refer to the `README.md` files in each folder.
