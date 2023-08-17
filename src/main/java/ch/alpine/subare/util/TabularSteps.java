// code by jph
package ch.alpine.subare.util;

import java.util.List;

import ch.alpine.subare.api.DiscreteModel;
import ch.alpine.subare.api.SampleModel;
import ch.alpine.subare.api.StepDigest;
import ch.alpine.subare.api.StepRecord;

/** only suitable for models with all states as start states... */
public enum TabularSteps {
  ;
  /** parallel processing of batch of steps by given {@link StepDigest}s
   * 
   * @param discreteModel
   * @param sampleModel
   * @param stepDigest */
  public static void batch(DiscreteModel discreteModel, SampleModel sampleModel, StepDigest... stepDigest) {
    List<StepDigest> list = List.of(stepDigest);
    TabularStepBatch tabularStepBatch = new TabularStepBatch(discreteModel, sampleModel);
    while (tabularStepBatch.hasNext()) {
      StepRecord stepRecord = tabularStepBatch.next();
      list.stream().parallel().forEach(_stepDigest -> _stepDigest.digest(stepRecord));
    }
  }
}
