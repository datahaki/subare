// code by jph
package ch.alpine.subare.net.api;

import java.util.List;
import java.util.random.RandomGenerator;

import ch.alpine.subare.net.opt.NetOptimizers;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.Throw;

public interface NetLayer {
  /** @param x
   * @return forward of x */
  Tensor forward(Tensor x);

  /** @param X
   * @return forward of entire batch */
  Tensor forwardBatch(Tensor X);

  /** ErrorThe element-wise or raw difference between a model's
   * prediction and the ground truth target (e.g., error = y - target ).
   * It is vector-valued or tensor-valued per sample.
   * 
   * @param target
   * @return */
  default NetLoss error(Tensor target) {
    throw new UnsupportedOperationException();
  }

  /** @param gradOutput
   * @return */
  Tensor back(Tensor gradOutput);

  /** back propagation of gradient of entire batch
   * 
   * @param gradOutput
   * @return */
  Tensor backBatch(Tensor gradOutput);

  /** visits all members
   * useful for pointwise operations such as
   * {@link #initialize(RandomGenerator)}
   * {@link #parameters()}
   * {@link #gradient(boolean)}
   * {@link #increment(boolean, Tensor, Scalar)}
   * {@link #reg_loss(boolean)}
   * 
   * @return all net layers directly harbored by this instance */
  default List<NetLayer> nested() {
    return List.of();
  }

  /** Parametric Layers:
   * Layers that contain internal variables (weights and biases) that are adjusted and
   * optimized during training (e.g., Dense / Linear layers, Convolutional layers).
   * 
   * Layers whose parameters receive gradients during backpropagation and are updated
   * by the optimizer. */
  default void initialize(RandomGenerator randomGenerator) {
    // do nothing
  }

  /** @return */
  default Tensor parameters() {
    return Tensors.unmodifiableEmpty();
  }

  /** @param decayInLoss whether to include the gradient from the L2 (!) regularization,
   * i.e. "weight decay", in the gradient descent
   * @return */
  default Tensor gradient(boolean decayInLoss) {
    return Tensors.unmodifiableEmpty();
  }

  /** @param decayInLoss whether the gradient from the L2 (!) regularization was considered
   * in the gradient descent
   * @param delta
   * @param eta */
  default void increment(boolean decayInLoss, Tensor delta, Scalar eta) {
    Throw.unless(Tensors.isEmpty(delta));
  }

  /** note:
   * for certain {@link NetOptimizer} the weight decay does not
   * show in the loss function
   * 
   * @return contribution to loss by weight regularization
   * @see NetOptimizers */
  default Scalar reg_loss(boolean decayInLoss) {
    return RealScalar.ZERO;
  }
}
