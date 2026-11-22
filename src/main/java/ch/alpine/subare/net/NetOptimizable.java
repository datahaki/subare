// code by jph
package ch.alpine.subare.net;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public interface NetOptimizable {
  /** @param decayInLoss whether to include the gradient from the L2 (!) regularization,
   * i.e. "weight decay", in the gradient descent
   * @return */
  Tensor gradient(boolean decayInLoss);

  /** @param decayInLoss whether the gradient from the L2 (!) regularization was considered
   * in the gradient descent
   * @param delta
   * @param eta */
  void step(boolean decayInLoss, Tensor delta, Scalar eta);
}
