// code by jph
package ch.alpine.subare.util;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;

import ch.alpine.subare.api.DequeDigest;
import ch.alpine.subare.api.EpisodeInterface;
import ch.alpine.subare.api.MonteCarloInterface;
import ch.alpine.subare.api.StepRecord;

public abstract class DequeExploringStarts extends AbstractExploringStarts {
  private final int nstep;
  private final List<DequeDigest> list;

  protected DequeExploringStarts( //
      MonteCarloInterface monteCarloInterface, int nstep, DequeDigest... dequeDigest) {
    super(monteCarloInterface);
    this.nstep = nstep;
    list = List.of(dequeDigest);
    nextBatch();
  }

  @Override
  public final void protected_nextEpisode(EpisodeInterface episodeInterface) {
    Deque<StepRecord> deque = new ArrayDeque<>();
    while (episodeInterface.hasNext()) {
      final StepRecord stepInterface = episodeInterface.step();
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
