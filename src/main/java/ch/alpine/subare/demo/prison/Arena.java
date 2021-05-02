// code by jph
package ch.alpine.subare.demo.prison;

import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;

/** Julian's idea: Prisoners' Dilemma */
/* package */ enum Arena {
  ;
  /** rewards average at 2 */
  static final Tensor R0 = Tensors.matrixInt(new int[][] { //
      { -1, 2 }, //
      { -2, 1 } }).multiply(RationalScalar.HALF);
}
