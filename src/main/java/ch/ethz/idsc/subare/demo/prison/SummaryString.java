// code by jph
package ch.ethz.idsc.subare.demo.prison;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.sca.Round;
import ch.ethz.idsc.subare.ch02.Agent;

/* package */ enum SummaryString {
  ;
  public static String of(Agent agent) {
    int rnd = agent.getRandomizedDecisionCount();
    Scalar avg = Round._3.apply(agent.getRewardAverage());
    return String.format("%25s  %s  %5d RND", //
        agent.toString(), avg, rnd);
  }
}
