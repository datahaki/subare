// code by jph
package ch.alpine.subare.net.los;

import ch.alpine.tensor.Tensor;

public interface LossFunction {
  NetLoss compute(Tensor prediction, Tensor target);

  NetLoss computeBatch(Tensor prediction, Tensor target);
}
