// code by jph
package ch.alpine.subare.core.util;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import ch.alpine.subare.core.api.DiscreteModel;
import ch.alpine.subare.core.api.SampleModel;
import ch.alpine.subare.core.api.StepRecord;
import ch.alpine.subare.util.Index;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

/** only suitable for models with all states as start states */
/* package */ class TabularStepBatch {
  private final SampleModel sampleModel;
  private final List<Tensor> list;
  private int index = 0;

  /** constructor generates randomized batch of steps
   * collection consists of all state-action pairs including terminal states */
  public TabularStepBatch(DiscreteModel discreteModel, SampleModel sampleModel) {
    this.sampleModel = sampleModel;
    Index index = DiscreteUtils.build(discreteModel, discreteModel.states());
    list = index.keys().stream().collect(Collectors.toList());
    Collections.shuffle(list);
  }

  public boolean hasNext() {
    return index < list.size();
  }

  public StepRecord next() {
    Tensor key = list.get(index);
    ++index;
    return step(key.get(0), key.get(1));
  }

  private StepRecord step(Tensor state, Tensor action) {
    Tensor next = sampleModel.move(state, action);
    Scalar reward = sampleModel.reward(state, action, next);
    return new StepRecord(state, action, reward, next);
  }
}
