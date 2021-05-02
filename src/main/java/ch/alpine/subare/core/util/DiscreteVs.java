// code by jph
package ch.alpine.subare.core.util;

import java.io.Serializable;
import java.util.stream.Stream;

import ch.alpine.subare.core.VsInterface;
import ch.alpine.subare.util.Index;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.TensorRuntimeException;
import ch.alpine.tensor.alg.Array;

public class DiscreteVs implements VsInterface, DiscreteValueFunction, Serializable {
  /** initializes all state value to zero
   * 
   * @param discreteModel
   * @return */
  public static DiscreteVs build(Tensor states) {
    return build(states, Array.zeros(states.length()));
  }

  public static DiscreteVs build(Tensor states, Tensor values) {
    return new DiscreteVs(Index.build(states), values);
  }

  /***************************************************/
  private final Index index;
  private final Tensor values;

  /** @param index
   * @param values */
  public DiscreteVs(Index index, Tensor values) {
    if (index.size() != values.length())
      throw TensorRuntimeException.of(index.keys(), values);
    this.index = index;
    this.values = values;
  }

  @Override // from VsInterface
  public Scalar value(Tensor state) {
    return values.Get(index.of(state));
  }

  @Override // from VsInterface
  public void increment(Tensor state, Scalar delta) {
    values.set(delta::add, index.of(state));
  }

  public void assign(Tensor state, Scalar value) {
    values.set(value, index.of(state));
  }

  @Override // from VsInterface
  public DiscreteVs copy() {
    return new DiscreteVs(index, values.copy());
  }

  @Override // from VsInterface
  public DiscreteVs discounted(Scalar gamma) {
    return new DiscreteVs(index, values.multiply(gamma));
  }

  /**************************************************/
  @Override // from DiscreteValueFunction
  public Tensor keys() {
    return index.keys();
  }

  @Override // from DiscreteValueFunction
  public Tensor values() {
    return values.unmodifiable();
  }

  @Override // from DiscreteValueFunction
  public DiscreteVs create(Stream<? extends Tensor> stream) {
    return new DiscreteVs(index, Tensor.of(stream));
  }
}
