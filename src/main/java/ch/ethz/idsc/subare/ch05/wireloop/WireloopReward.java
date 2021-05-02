// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.ethz.idsc.subare.core.RewardInterface;

/** collection of different cost functions */
public interface WireloopReward extends RewardInterface {
  /** steps don't cost anything
   * 
   * @return constant zero */
  static WireloopReward freeSteps() {
    return (s, a, n) -> RealScalar.ZERO;
  }

  /** steps are more expensive than reward along x
   * 
   * @return constant zero */
  static WireloopReward constantCost() {
    return (s, a, n) -> RealScalar.of(-1.4); // -1.2
  }

  /***************************************************/
  static Scalar id_x(Tensor state) {
    return state.Get(0);
  }
}
