// code by jph
package ch.alpine.subare.core.util;

import java.util.List;

import ch.alpine.subare.core.api.EpisodeInterface;
import ch.alpine.subare.core.api.MonteCarloInterface;
import ch.alpine.subare.core.api.StepDigest;
import ch.alpine.subare.core.api.StepRecord;

public abstract class StepExploringStarts extends AbstractExploringStarts {
  private final List<StepDigest> list;

  protected StepExploringStarts(MonteCarloInterface monteCarloInterface, StepDigest... dequeDigest) {
    super(monteCarloInterface);
    list = List.of(dequeDigest);
    nextBatch();
  }

  @Override
  public final void protected_nextEpisode(EpisodeInterface episodeInterface) {
    while (episodeInterface.hasNext()) {
      StepRecord stepInterface = episodeInterface.step();
      list.stream().parallel() //
          .forEach(_dequeDigest -> _dequeDigest.digest(stepInterface));
    }
  }
}
