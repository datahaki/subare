// code by jph
package ch.alpine.subare.book.ch05.wireloop;

import ch.alpine.subare.api.RewardInterface;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

/** collection of different cost functions */
public interface WireloopReward extends RewardInterface {
  /** steps don't cost anything
   * 
   * @return constant zero */
  static WireloopReward freeSteps() {
    return (_, _, _) -> RealScalar.ZERO;
  }

  /** steps are more expensive than reward along x
   * 
   * @return constant zero */
  static WireloopReward constantCost() {
    return (_, _, _) -> RealScalar.of(-1.4); // -1.2
  }

  // ---
  static Scalar id_x(Tensor state) {
    return state.Get(0);
  }
}
