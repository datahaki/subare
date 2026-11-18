// code by jph
package ch.alpine.subare.net.api;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public interface NetRegularization {
  Scalar penalty(boolean decayInLoss, Tensor weights);

  Tensor gradient(boolean decayInLoss, Tensor weights);

  Tensor sneakDecay(boolean decayInLoss, Tensor weights);
}
