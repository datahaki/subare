// code by jph
package ch.alpine.subare.net.api;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public interface NetRegularization {
  Scalar penalty(boolean weightDecay, Tensor weights);

  Tensor gradient(boolean weightDecay, Tensor weights);

  Tensor sneakDecay(boolean weightDecay, Tensor weights);
}
