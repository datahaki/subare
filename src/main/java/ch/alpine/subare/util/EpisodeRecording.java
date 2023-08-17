// code by jph
package ch.alpine.subare.util;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import ch.alpine.subare.api.EpisodeInterface;
import ch.alpine.subare.api.StepRecord;

/** class steps through a given episode and stores the steps for one or multiple replays */
public class EpisodeRecording {
  private final List<StepRecord> list = new LinkedList<>();

  public EpisodeRecording(EpisodeInterface episodeInterface) {
    while (episodeInterface.hasNext()) {
      StepRecord stepRecord = episodeInterface.step();
      list.add(stepRecord);
    }
  }

  public EpisodeInterface replay() {
    return new EpisodeInterface() {
      final Iterator<StepRecord> iterator = list.iterator();

      @Override
      public StepRecord step() {
        return iterator.next();
      }

      @Override
      public boolean hasNext() {
        return iterator.hasNext();
      }
    };
  }
}
