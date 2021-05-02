// code by jph
package ch.alpine.subare.core.util;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;

import ch.alpine.subare.core.DequeDigest;
import ch.alpine.subare.core.EpisodeInterface;
import ch.alpine.subare.core.MonteCarloInterface;
import ch.alpine.subare.core.StepInterface;

public abstract class DequeExploringStarts extends AbstractExploringStarts {
  private final int nstep;
  private final List<DequeDigest> list;

  public DequeExploringStarts( //
      MonteCarloInterface monteCarloInterface, int nstep, DequeDigest... dequeDigest) {
    super(monteCarloInterface);
    this.nstep = nstep;
    list = Arrays.asList(dequeDigest);
    nextBatch();
  }

  @Override
  public final void protected_nextEpisode(EpisodeInterface episodeInterface) {
    Deque<StepInterface> deque = new ArrayDeque<>();
    while (episodeInterface.hasNext()) {
      final StepInterface stepInterface = episodeInterface.step();
      deque.add(stepInterface);
      if (deque.size() == nstep) { // never true, if nstep == 0
        list.stream().parallel() //
            .forEach(_dequeDigest -> _dequeDigest.digest(deque));
        deque.poll();
      }
    }
    while (!deque.isEmpty()) {
      list.stream().parallel() //
          .forEach(_dequeDigest -> _dequeDigest.digest(deque));
      deque.poll();
    }
  }
}
