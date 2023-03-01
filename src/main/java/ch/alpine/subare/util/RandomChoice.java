// code by jph
package ch.alpine.subare.util;

import java.security.SecureRandom;
import java.util.List;
import java.util.random.RandomGenerator;

import ch.alpine.tensor.Tensor;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/RandomChoice.html">RandomChoice</a> */
public enum RandomChoice {
  ;
  private static final RandomGenerator RANDOM = new SecureRandom();

  /** @param tensor
   * @return random entry of tensor
   * @throws Exception if given tensor is empty */
  @SuppressWarnings("unchecked")
  public static <T extends Tensor> T of(Tensor tensor) {
    return (T) tensor.get(RANDOM.nextInt(tensor.length()));
  }

  // ---
  /** @param list
   * @param random
   * @return random entry of list
   * @throws Exception if given list is empty */
  public static <T> T of(List<T> list, RandomGenerator random) {
    return list.get(random.nextInt(list.size()));
  }

  /** @param list
   * @return */
  public static <T> T of(List<T> list) {
    return of(list, RANDOM);
  }
}
