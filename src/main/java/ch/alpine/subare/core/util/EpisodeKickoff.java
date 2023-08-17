// code by jph
package ch.alpine.subare.core.util;

import java.util.ArrayDeque;
import java.util.Random;
import java.util.random.RandomGenerator;

import ch.alpine.subare.core.api.EpisodeInterface;
import ch.alpine.subare.core.api.MonteCarloInterface;
import ch.alpine.subare.core.api.Policy;
import ch.alpine.subare.core.mc.MonteCarloEpisode;
import ch.alpine.tensor.Tensor;

public enum EpisodeKickoff {
  ;
  private static final RandomGenerator RANDOM = new Random();

  public static EpisodeInterface single(MonteCarloInterface monteCarloInterface, Policy policy) {
    Tensor starts = monteCarloInterface.startStates();
    Tensor start = starts.get(RANDOM.nextInt(starts.length()));
    return single(monteCarloInterface, policy, start);
  }

  public static EpisodeInterface single(MonteCarloInterface monteCarloInterface, Policy policy, Tensor start) {
    if (monteCarloInterface.isTerminal(start))
      throw new IllegalStateException();
    return new MonteCarloEpisode(monteCarloInterface, policy, start, new ArrayDeque<>());
  }
}
