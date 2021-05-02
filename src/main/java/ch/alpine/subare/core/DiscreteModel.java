// code by jph
package ch.alpine.subare.core;

import ch.alpine.tensor.Scalar;

public interface DiscreteModel extends StateActionModel {
  /** @return discount factor in the interval [0, 1] */
  Scalar gamma();
}
