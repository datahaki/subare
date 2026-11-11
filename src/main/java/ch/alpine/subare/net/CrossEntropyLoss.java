// code by jph
package ch.alpine.subare.net;

import java.util.stream.IntStream;

import ch.alpine.tensor.Rational;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Unprotect;
import ch.alpine.tensor.alg.UnitVector;
import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.io.Primitives;
import ch.alpine.tensor.red.Max;
import ch.alpine.tensor.red.Mean;
import ch.alpine.tensor.sca.exp.Log;

public record CrossEntropyLoss(Scalar loss, Tensor gradOutput) implements NetError {
  static final ScalarUnaryOperator MAX = Max.function(RealScalar.of(1e-15));

  /** @param interCache
   * @param target scalar or vector of integer values between 0 and interCache.length
   * @return */
  public static NetError of(Tensor interCache, Tensor target) {
    Tensor onehot;
    Scalar loss;
    if (target instanceof Scalar) {
      int k = Scalars.intValueExact((Scalar) target);
      loss = Log.FUNCTION.apply(MAX.apply(interCache.Get(k))).negate();
      // TODO not efficient to build unit vector explicitly
      int cc = interCache.length();
      onehot = UnitVector.of(cc, k); // one-hot target
      Tensor error = interCache.subtract(onehot);
      return new CrossEntropyLoss(loss, error);
    }
    int[] index = Primitives.toIntArray(target);
    int batch_size = index.length;
    // TODO use sparse array for fun!
    int cc = Unprotect.dimension1Hint(interCache);
    onehot = Tensor.of(IntStream.range(0, batch_size) //
        .mapToObj(i -> UnitVector.of(cc, index[i])));
    loss = Mean.ofVector(Tensor.of(IntStream.range(0, batch_size) //
        .mapToObj(i -> interCache.Get(i, index[i])) //
        .map(MAX).map(Log.FUNCTION))).negate();
    Tensor error = interCache.subtract(onehot);
    Tensor gradOutput = error.multiply(Rational.of(1, batch_size));
    return new CrossEntropyLoss(loss, gradOutput);
  }
}
