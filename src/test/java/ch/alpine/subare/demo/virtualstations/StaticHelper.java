// code by fluric
package ch.alpine.subare.demo.virtualstations;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Append;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.alg.Join;

/* package */ enum StaticHelper {
  ;
  /** returns the tensor of all possible binary combinations in a vector of size length
   * 
   * @param length
   * @param prefixes
   * @return */
  public static Tensor binaryVectors(int length, Tensor prefixes) {
    if (length == 0)
      return prefixes;
    if (Tensors.isEmpty(prefixes))
      return binaryVectors(length - 1, Tensors.of(Tensors.vector(1), Tensors.vector(0)));
    Tensor extension = Tensors.empty();
    for (Tensor prefix : prefixes) {
      extension.append(Append.of(prefix, RealScalar.ONE));
      extension.append(Append.of(prefix, RealScalar.ZERO));
    }
    return binaryVectors(length - 1, extension);
  }

  public static Tensor zeroVectors(int length, Tensor prefixes) {
    if (Tensors.isEmpty(prefixes))
      return Tensors.of(Array.zeros(length));
    return Tensor.of(prefixes.stream().map(v -> Join.of(v, Array.zeros(length))));
  }
}
