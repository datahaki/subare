// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.TensorRuntimeException;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.EmpiricalDistribution;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.StateActionModel;
import ch.ethz.idsc.subare.util.Index;

/** the term "equiprobable" appears in Exercise 4.1 */
public class EquiprobablePolicy implements Policy {
  /** @param stateActionModel
   * @return */
  public static Policy create(StateActionModel stateActionModel) {
    return new EquiprobablePolicy(Objects.requireNonNull(stateActionModel));
  }

  // ---
  private final StateActionModel stateActionModel;
  private final Map<Tensor, Index> map = new HashMap<>();

  private EquiprobablePolicy(StateActionModel stateActionModel) {
    this.stateActionModel = stateActionModel;
  }

  @Override
  public synchronized Scalar probability(Tensor state, Tensor action) {
    Index index = map.get(state);
    if (Objects.isNull(index)) {
      index = Index.build(stateActionModel.actions(state));
      map.put(state, index);
    }
    if (index.containsKey(action)) // alternatively return 0
      return RationalScalar.of(1, index.size());
    throw TensorRuntimeException.of(state, action); // action invalid
  }

  @Override
  public Distribution getDistribution(Tensor state) {
    Tensor pdf = Tensor.of(stateActionModel.actions(state).stream() //
        .map(action -> probability(state, action)));
    return EmpiricalDistribution.fromUnscaledPDF(pdf);
  }
}
