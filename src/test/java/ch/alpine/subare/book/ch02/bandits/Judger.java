// code by jph
package ch.alpine.subare.book.ch02.bandits;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ch.alpine.subare.book.ch02.Agent;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Round;

/* package */ class Judger {
  private final Bandits bandit;
  private final Map<Agent, Tensor> map = new HashMap<>();

  Judger(Bandits bandit, Agent... agents) {
    this.bandit = bandit;
    Arrays.stream(agents).forEach(agent -> map.put(agent, Tensors.empty()));
  }

  void play() {
    Tensor tensor = bandit.pullAll();
    for (Agent agent : map.keySet()) {
      int k = agent.takeAction();
      Scalar value = tensor.Get(k);
      agent.feedback(k, value);
      map.get(agent).append(agent.getRewardTotal());
    }
  }

  void ranking() {
    List<Agent> list = new ArrayList<>(map.keySet());
    list.sort((a1, a2) -> Scalars.compare(a1.getRewardTotal(), a2.getRewardTotal()));
    Clip clip = bandit.clip();
    for (Agent agent : list) {
      Scalar s = clip.rescale(agent.getRewardTotal()).multiply(RealScalar.of(100));
      System.out.printf("%25s%5s %%%8s RND%n", //
          agent, Round.FUNCTION.apply(s), "" + agent.getRandomizedDecisionCount());
    }
  }

  Map<Agent, Tensor> map() {
    return Collections.unmodifiableMap(map);
  }
}
