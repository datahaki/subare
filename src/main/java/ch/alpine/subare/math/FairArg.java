// code by jph
package ch.alpine.subare.math;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.function.BinaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.ext.RandomChoice;
import ch.alpine.tensor.red.Max;
import ch.alpine.tensor.red.Min;

/** ArgMax returns the index of the first element in the input sequence, which is equal to the max value. */
public class FairArg implements Serializable {
  /** @param tensor
   * @return
   * @throws Exception if tensor is empty, or a scalar */
  public static FairArg max(Tensor tensor) {
    return new FairArg(build(Max::of, tensor));
  }

  /** @param tensor
   * @return
   * @throws Exception if tensor is empty, or a scalar */
  public static FairArg min(Tensor tensor) {
    return new FairArg(build(Min::of, tensor));
  }

  private static List<Integer> build(BinaryOperator<Tensor> binaryOperator, Tensor tensor) {
    Tensor value = tensor.stream().reduce(binaryOperator).get();
    return IntStream.range(0, tensor.length()) //
        .filter(index -> tensor.get(index).equals(value)) //
        .boxed() //
        .collect(Collectors.toList());
  }

  // ---
  private final List<Integer> list;

  private FairArg(List<Integer> list) {
    this.list = list;
  }

  public int nextRandomIndex() {
    return RandomChoice.of(list);
  }

  public boolean isUnique() {
    return optionsCount() == 1;
  }

  public int optionsCount() {
    return list.size();
  }

  public List<Integer> options() {
    return Collections.unmodifiableList(list);
  }
}
