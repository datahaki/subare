// code by jph
package ch.alpine.subare.util;

import java.io.Serializable;
import java.util.stream.IntStream;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.red.Max;
import ch.alpine.tensor.sca.Chop;

/** RobustArgMax accounts for entries that are numerically close to the maximum and
 * returns the first such close match.
 * 
 * @param chop that performs proximity check to the max via {@link Chop#isClose(Tensor, Tensor)} */
public record RobustArgMax(Chop chop) implements Serializable {
  /** @param vector
   * @return indices of entries that are close to the maximum entry in vector
   * @throws Exception if vector is empty, or not a tensor of rank 1 */
  public IntStream options(Tensor vector) {
    Tensor max = vector.stream().reduce(Max::of).get();
    return IntStream.range(0, vector.length()) //
        .filter(index -> chop.isClose(vector.get(index), max));
  }

  /** in the spirit of ArgMax which returns the first of equally maximal indices.
   * 
   * @param vector
   * @return first index that is epsilon close to the maximum
   * @throws Exception if vector is empty, or not a tensor of rank 1 */
  public int of(Tensor vector) {
    return options(vector).findFirst().getAsInt();
  }
}
