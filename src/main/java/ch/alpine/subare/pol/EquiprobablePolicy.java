// code by jph
package ch.alpine.subare.pol;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import ch.alpine.subare.math.Index;
import ch.alpine.subare.mod.DiscreteModel;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Throw;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.d.CategoricalDistribution;

/** the term "equiprobable" appears in Exercise 4.1 */
public class EquiprobablePolicy implements Policy {
  /** @param discreteModel
   * @return */
  public static Policy create(DiscreteModel discreteModel) {
    return new EquiprobablePolicy(Objects.requireNonNull(discreteModel));
  }

  // ---
  private final DiscreteModel discreteModel;
  private final Map<Tensor, Index> map = new HashMap<>();

  private EquiprobablePolicy(DiscreteModel discreteModel) {
    this.discreteModel = discreteModel;
  }

  @Override
  public synchronized Scalar probability(Tensor state, Tensor action) {
    Index index = map.computeIfAbsent(state, s -> Index.build(discreteModel.actions(s)));
    // Index index = map.get(state);
    // if (Objects.isNull(index)) {
    // index = Index.build(stateActionModel.actions(state));
    // map.put(state, index);
    // }
    if (index.containsKey(action)) // alternatively return 0
      return Rational.of(1, index.size());
    throw new Throw(state, action); // action invalid
  }

  @Override
  public Distribution getDistribution(Tensor state) {
    Tensor pdf = Tensor.of(discreteModel.actions(state).stream() //
        .map(action -> probability(state, action)));
    return CategoricalDistribution.fromUnscaledPDF(pdf);
  }
}
