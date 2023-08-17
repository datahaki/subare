// code by jph
package ch.alpine.subare.util;

import java.util.ArrayDeque;
import java.util.Collections;
import java.util.List;
import java.util.Queue;
import java.util.stream.Collectors;

import ch.alpine.subare.api.EpisodeInterface;
import ch.alpine.subare.api.MonteCarloInterface;
import ch.alpine.subare.api.Policy;
import ch.alpine.subare.math.Index;
import ch.alpine.subare.mc.MonteCarloEpisode;
import ch.alpine.tensor.Tensor;

/* package */ class ExploringStartsBatch {
  private final MonteCarloInterface monteCarloInterface;
  /** list contains all starting start-action pairs, shuffled randomly */
  private final List<Tensor> list;
  private int index = 0;

  /* package */ ExploringStartsBatch(MonteCarloInterface monteCarloInterface) {
    this.monteCarloInterface = monteCarloInterface;
    Index index = DiscreteUtils.build(monteCarloInterface, monteCarloInterface.startStates());
    list = index.keys().stream().collect(Collectors.toList());
    Collections.shuffle(list);
  }

  /** @return true if call to nextEpisode is valid */
  public boolean hasNext() {
    return index < list.size();
  }

  /** @param policy
   * @return
   * @throws Exception if hasNext() == false */
  public EpisodeInterface nextEpisode(Policy policy) {
    Tensor key = list.get(index);
    Tensor state = key.get(0); // first state
    Tensor action = key.get(1); // first action
    if (monteCarloInterface.isTerminal(state)) // consistency check
      throw new IllegalStateException();
    Queue<Tensor> queue = new ArrayDeque<>();
    queue.add(action);
    ++index;
    return new MonteCarloEpisode(monteCarloInterface, policy, state, queue);
  }
}
