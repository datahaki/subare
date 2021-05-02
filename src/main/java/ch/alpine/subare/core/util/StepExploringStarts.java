// code by jph
package ch.alpine.subare.core.util;

import java.util.Arrays;
import java.util.List;

import ch.alpine.subare.core.EpisodeInterface;
import ch.alpine.subare.core.MonteCarloInterface;
import ch.alpine.subare.core.StepDigest;
import ch.alpine.subare.core.StepInterface;

public abstract class StepExploringStarts extends AbstractExploringStarts {
  private final List<StepDigest> list;

  public StepExploringStarts(MonteCarloInterface monteCarloInterface, StepDigest... dequeDigest) {
    super(monteCarloInterface);
    list = Arrays.asList(dequeDigest);
    nextBatch();
  }

  @Override
  public final void protected_nextEpisode(EpisodeInterface episodeInterface) {
    while (episodeInterface.hasNext()) {
      StepInterface stepInterface = episodeInterface.step();
      list.stream().parallel() //
          .forEach(_dequeDigest -> _dequeDigest.digest(stepInterface));
    }
  }
}
