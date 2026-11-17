package ch.alpine.subare.net.lay;

import ch.alpine.tensor.Tensor;

/** Parametric Layers:
 * Layers that contain internal variables (weights and biases) that are adjusted and
 * optimized during training (e.g., Dense / Linear layers, Convolutional layers).
 * 
 * Layers whose parameters receive gradients during backpropagation and are updated
 * by the optimizer. */
public interface ParametricLayer extends NetLayer {
  Tensor gradient();

  void increment(Tensor delta);

  Tensor getWb();
}
