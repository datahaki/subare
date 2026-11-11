// code by jph
package ch.alpine.subare.net;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public interface NetError {
  Scalar loss();

  Tensor gradOutput();
}
