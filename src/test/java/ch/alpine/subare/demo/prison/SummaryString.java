// code by jph
package ch.alpine.subare.demo.prison;

import ch.alpine.subare.book.ch02.Agent;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.sca.Round;

/* package */ enum SummaryString {
  ;
  public static String of(Agent agent) {
    int rnd = agent.getRandomizedDecisionCount();
    Scalar avg = Round._3.apply(agent.getRewardAverage());
    return String.format("%25s  %s  %5d RND", //
        agent, avg, rnd);
  }
}
