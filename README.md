# neural-networks-zero-to-hero

My solutions to the exercises from **Andrej Karpathy's** [Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0) series — specifically the micrograd video, *"The spelled-out intro to neural networks and backpropagation: building micrograd."*

## Contents

| File | What's in it |
| --- | --- |
| [micrograd.ipynb](micrograd.ipynb) | Follow-along notebook from the video: derivatives from first principles, the `Value` autograd engine, the expression graph, backpropagation, and a small MLP trained by gradient descent. |
| [micrograd_exercises.ipynb](micrograd_exercises.ipynb) | The exercise notebook set at the end of the video, worked through. |

## Exercises covered

1. **Analytical gradient** — derive and implement `gradf` for `f(a, b, c) = -a³ + sin(3b) - 1/c + b^2.5 - a^0.5`.
2. **Numerical gradient** — estimate the same gradient with the forward difference `(f(x+h) - f(x)) / h`.
3. **Symmetric derivative** — the centered difference `(f(x+h) - f(x-h)) / 2h`, and confirming it is a better approximation at the same step size `h`.
4. **Extending `Value`** — filling in the missing ops (`exp`, `log`, `__truediv__`, `__pow__`, `__neg__`, `__sub__`, …) so that a softmax plus negative log-likelihood loss backpropagates correctly.

## Running

```bash
pip install numpy matplotlib
jupyter notebook
```

Then open either notebook. No other dependencies — the whole autograd engine is written from scratch in the notebook.

## Credit

All exercises and the original micrograd design are Andrej Karpathy's: [karpathy/micrograd](https://github.com/karpathy/micrograd) · [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero). This repo is just my working through them.
