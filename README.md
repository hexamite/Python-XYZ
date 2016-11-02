First use a analytical method to estimate xyz coordinate (either with the three nearest beacons as in the code or do all combinations and then average) as outlined in https://en.wikipedia.org/wiki/Trilateration.
As a next step, use the Nonlinear Least Squares (NLSQ) Model as described in https://inside.mines.edu/~whereman/talks/TurgutOzal-11-Trilateration.pdf

please note: in the paper they use a linear method to approximate the first XYZ guess - that doesn't work in my setup: all my transmitters are roughly at the same hight which gives for a very bad matrix condition.
