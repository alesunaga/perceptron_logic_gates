# Perceptron Logic Gates

This repository contains Python code demonstrating the implementation and visualization of a Perceptron, a foundational model in machine learning, to simulate basic logic gates: AND, OR, and XOR.

## Project Overview

Perceptrons are the simplest form of artificial neural networks, capable of learning linear decision boundaries. This project illustrates how a single-layer Perceptron can accurately model linearly separable functions like AND and OR gates, while also highlighting its limitations when dealing with non-linearly separable functions like XOR.

The script performs the following:
* Defines input data and corresponding labels for AND, OR, and XOR logic gates.
* Trains a `Perceptron` classifier from `scikit-learn` for each gate.
* Visualizes the input data points using scatter plots.
* Evaluates the accuracy of the Perceptron for each gate.
* Generates heatmaps to visualize the decision function (hyperplane) of the trained Perceptron, showing how it classifies the entire input space.

## Why this project?

This project serves as an excellent educational tool for understanding:
* **The basics of Perceptrons:** How they learn and make classifications.
* **Linear Separability:** The concept that underlies the Perceptron's capabilities and limitations.
* **Logic Gates:** The fundamental building blocks of digital circuits, and how they can be represented computationally.
* **Data Visualization:** Using `matplotlib` to interpret machine learning model behavior through scatter plots and heatmaps.

As an educational technology coordinator and Makerspace manager, I believe in making complex topics accessible and engaging. This project connects theoretical AI concepts with practical applications, which aligns with my work in promoting digital transformation and active learning.

## Getting Started

### Prerequisites

To run this code, you'll need the following Python libraries:
* `scikit-learn`
* `matplotlib`
* `numpy`
* `codecademylib3_seaborn` (This specific library is often used in Codecademy's environment for plot styling. If you're running this outside of Codecademy, you might be able to remove this import or substitute it with `seaborn` for similar aesthetics, or simply run without it.)

You can install them using pip:

```bash
pip install scikit-learn matplotlib numpy
# If needed, for the styling specific to Codecademy's environment:
# pip install codecademylib3_seaborn
