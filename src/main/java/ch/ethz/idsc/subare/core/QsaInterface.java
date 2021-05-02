// code by jph
package ch.ethz.idsc.subare.core;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public interface QsaInterface {
  /** @param state
   * @param action
   * @return value of state-action pair */
  Scalar value(Tensor state, Tensor action);

  /** map state-action pair to given value
   * 
   * @param state
   * @param action
   * @param value
   * @throws Exception if assign operation is not supported */
  void assign(Tensor state, Tensor action, Scalar value);

  /** @return */
  QsaInterface copy();
}
