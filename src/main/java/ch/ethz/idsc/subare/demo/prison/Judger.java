// code by jph
package ch.ethz.idsc.subare.demo.prison;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.sca.Chop;
import ch.ethz.idsc.subare.ch02.Agent;

/* package */ class Judger {
  private final Tensor reward;
  private final Agent a1;
  private final Agent a2;

  Judger(Tensor r1, Agent a1, Agent a2) {
    this.reward = r1;
    this.a1 = a1;
    this.a2 = a2;
  }

  void play() {
    int A1 = a1.takeAction();
    int A2 = a2.takeAction();
    a1.feedback(A1, reward.Get(A1, A2));
    a2.feedback(A2, reward.Get(A2, A1));
  }

  /** @return tensor of rewards averaged over number of actions */
  Tensor ranking() {
    Chop.NONE.requireClose(a1.getCount(), a2.getCount());
    return Tensors.of(a1.getRewardTotal(), a2.getRewardTotal()).divide(a1.getCount());
  }
}
