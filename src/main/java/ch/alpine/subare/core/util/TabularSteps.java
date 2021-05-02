// code by jph
package ch.alpine.subare.core.util;

import java.util.Arrays;
import java.util.List;

import ch.alpine.subare.core.DiscreteModel;
import ch.alpine.subare.core.SampleModel;
import ch.alpine.subare.core.StepDigest;
import ch.alpine.subare.core.StepInterface;

/** only suitable for models with all states as start states... */
public enum TabularSteps {
  ;
  /** parallel processing of batch of steps by given {@link StepDigest}s
   * 
   * @param discreteModel
   * @param sampleModel
   * @param stepDigest */
  public static void batch(DiscreteModel discreteModel, SampleModel sampleModel, StepDigest... stepDigest) {
    List<StepDigest> list = Arrays.asList(stepDigest);
    TabularStepBatch tabularStepBatch = new TabularStepBatch(discreteModel, sampleModel);
    while (tabularStepBatch.hasNext()) {
      StepInterface stepInterface = tabularStepBatch.next();
      list.stream().parallel().forEach(_stepDigest -> _stepDigest.digest(stepInterface));
    }
  }
}
