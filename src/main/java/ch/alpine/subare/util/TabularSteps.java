// code by jph
package ch.alpine.subare.util;

import java.util.List;

import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.mod.TabularModel;
import ch.alpine.subare.pol.StepDigest;

/** only suitable for models with all states as start states... */
public enum TabularSteps {
  ;
  /** parallel processing of batch of steps by given {@link StepDigest}s
   * 
   * @param tabularModel
   * @param sampleModel
   * @param stepDigest */
  public static void batch(TabularModel tabularModel, StepDigest... stepDigest) {
    List<StepDigest> list = List.of(stepDigest);
    TabularStepBatch tabularStepBatch = new TabularStepBatch(tabularModel);
    while (tabularStepBatch.hasNext()) {
      StepRecord stepRecord = tabularStepBatch.next();
      list.stream().parallel().forEach(_stepDigest -> _stepDigest.digest(stepRecord));
    }
  }
}
