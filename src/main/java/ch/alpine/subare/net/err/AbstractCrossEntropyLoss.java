// code by jph
package ch.alpine.subare.net.err;

import java.io.Serializable;

import ch.alpine.subare.net.LossFunction;
import ch.alpine.subare.net.NetLoss;
import ch.alpine.subare.net.lay.ArgMaxLayer;
import ch.alpine.tensor.Tensor;

public abstract class AbstractCrossEntropyLoss implements LossFunction, Serializable {
  /** @param _p is discarded, probabilities() will be queried instead
   * @param target is a scalar of integer value between 0, 1, 2, and "number of classes"
   * @return */
  @Override
  public NetLoss compute(Tensor _p, Tensor target) {
    return CrossEntropyLoss.SCALAR.compute(probabilities(), target);
  }

  /** @param _p is discarded, probabilities() will be queried instead
   * @param target vector of integers
   * @return */
  @Override
  public NetLoss computeBatch(Tensor _p, Tensor target) {
    return CrossEntropyLoss.SCALAR.computeBatch(probabilities(), target);
  }

  /** @return logits provided by {@link ArgMaxLayer} */
  protected abstract Tensor probabilities();

  @Override
  public String toString() {
    return CrossEntropyLoss.SCALAR.toString();
  }
}
