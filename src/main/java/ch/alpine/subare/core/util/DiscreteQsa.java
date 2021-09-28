// code by jph
package ch.alpine.subare.core.util;

import java.io.Serializable;
import java.util.stream.Stream;

import ch.alpine.subare.core.MonteCarloInterface;
import ch.alpine.subare.core.QsaInterface;
import ch.alpine.subare.core.StateActionModel;
import ch.alpine.subare.util.Index;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.ext.Integers;
import ch.alpine.tensor.red.Max;
import ch.alpine.tensor.red.Min;

public class DiscreteQsa implements QsaInterface, DiscreteValueFunction, Serializable {
  /** @param stateActionModel
   * @return qsa with q(s, a) == 0 for all state-action pairs */
  public static DiscreteQsa build(StateActionModel stateActionModel) {
    Index index = DiscreteUtils.build(stateActionModel, stateActionModel.states());
    return new DiscreteQsa(index, Array.zeros(index.size()));
  }

  public static DiscreteQsa build(MonteCarloInterface monteCarloInterface, Scalar init) {
    Index index = DiscreteUtils.build(monteCarloInterface, monteCarloInterface.states());
    return new DiscreteQsa(index, Tensors.vector(i -> monteCarloInterface.isTerminal(index.get(i).get(0)) //
        ? RealScalar.ZERO
        : init, index.size()));
  }

  // ---
  private final Index index;
  private final Tensor values;

  private DiscreteQsa(Index index, Tensor values) {
    Integers.requireEquals(index.size(), values.length());
    this.index = index;
    this.values = values;
  }

  @Override // from QsaInterface
  public Scalar value(Tensor state, Tensor action) {
    return values.Get(index.of(StateAction.key(state, action)));
  }

  @Override // from QsaInterface
  public void assign(Tensor state, Tensor action, Scalar value) {
    values.set(value, index.of(StateAction.key(state, action)));
  }

  @Override // from QsaInterface
  public DiscreteQsa copy() {
    return new DiscreteQsa(index, values.copy());
  }

  // ---
  @Override // from DiscreteValueFunction
  public Tensor keys() {
    return index.keys();
  }

  @Override // from DiscreteValueFunction
  public Tensor values() {
    return values.unmodifiable();
  }

  @Override // from DiscreteValueFunction
  public DiscreteQsa create(Stream<? extends Tensor> stream) {
    return new DiscreteQsa(index, Tensor.of(stream));
  }

  // ---
  // TODO strictly speaking, the presence of these functions here is not requires
  public Scalar getMin() {
    return values.stream().map(Scalar.class::cast).reduce(Min::of).orElseThrow();
  }

  public Scalar getMax() {
    return values.stream().map(Scalar.class::cast).reduce(Max::of).orElseThrow();
  }

  public int size() {
    return index.size();
  }
}
