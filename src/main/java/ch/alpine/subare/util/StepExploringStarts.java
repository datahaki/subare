// code by jph
package ch.alpine.subare.util;

import java.util.List;

import ch.alpine.subare.api.EpisodeInterface;
import ch.alpine.subare.api.MonteCarloInterface;
import ch.alpine.subare.api.StepDigest;
import ch.alpine.subare.api.StepRecord;

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
      StepRecord stepRecord = episodeInterface.step();
      list.stream().parallel() //
          .forEach(_dequeDigest -> _dequeDigest.digest(stepRecord));
    }
  }
}
