// code by jph
package ch.alpine.subare.net.api;

import java.util.random.RandomGenerator;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

/** Parametric Layers:
 * Layers that contain internal variables (weights and biases) that are adjusted and
 * optimized during training (e.g., Dense / Linear layers, Convolutional layers).
 * 
 * Layers whose parameters receive gradients during backpropagation and are updated
 * by the optimizer. */
public interface ParametricLayer extends NetLayer {
  void initialize(RandomGenerator randomGenerator);

  /** @return */
  Tensor parameters();

  /** @param weightDecay whether to include the gradient from the L2 (!) regularization,
   * i.e. "weight decay", in the gradient descent
   * @return */
  Tensor gradient(boolean weightDecay);

  /** @param weightDecay whether the gradient from the L2 (!) regularization was considered
   * in the gradient descent
   * @param delta
   * @param eta */
  void increment(boolean weightDecay, Tensor delta, Scalar eta);

  /** @return contribution to loss by weight regularization */
  Scalar reg_loss(boolean weightDecay);
}
