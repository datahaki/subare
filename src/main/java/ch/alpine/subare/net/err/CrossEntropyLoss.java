// code by jph
package ch.alpine.subare.net.err;

import java.io.Serializable;
import java.util.stream.IntStream;

import ch.alpine.subare.net.NetLoss;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.Unprotect;
import ch.alpine.tensor.alg.UnitVector;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.red.Max;
import ch.alpine.tensor.red.Mean;
import ch.alpine.tensor.sca.Sign;
import ch.alpine.tensor.sca.exp.Log;

/** Loss == Cross Entropy divided by batch_size */
public record CrossEntropyLoss(Scalar loss, Tensor gradOutput) implements NetLoss, Serializable {
  static final ScalarUnaryOperator MAX = Max.function(RealScalar.of(1e-15));

  /** @param interCache non-negative probabilities
   * @param target
   * vector of length 1, or matrix of dimensions n x 1
   * of integer values between 0 and interCache.length
   * @return */
  public static NetLoss of(Tensor interCache, Tensor target) {
    Tensor onehot;
    Scalar loss;
    if (VectorQ.of(interCache)) {
      VectorQ.requireLength(target, 1);
      int k = Scalars.intValueExact(target.Get(0));
      loss = safeLog(interCache.Get(k)).negate();
      // TODO not efficient to build unit vector explicitly
      int cc = interCache.length();
      onehot = UnitVector.of(cc, k); // one-hot target
      Tensor error = interCache.subtract(onehot);
      return new CrossEntropyLoss(loss, error);
    }
    int batch_size = target.length();
    int[] index = target.stream() //
        .mapToInt(v -> Scalars.intValueExact(v.Get(0))) //
        .toArray();
    // TODO use sparse array for fun!
    int cc = Unprotect.dimension1Hint(interCache);
    onehot = Tensors.vector(i -> UnitVector.of(cc, index[i]), batch_size);
    loss = (Scalar) Mean.of(IntStream.range(0, batch_size) //
        .mapToObj(i -> interCache.Get(i, index[i])) //
        .map(CrossEntropyLoss::safeLog)).negate();
    Tensor error = interCache.subtract(onehot);
    Tensor gradOutput = error.multiply(Rational.of(1, batch_size));
    return new CrossEntropyLoss(loss, gradOutput);
  }

  private static Scalar safeLog(Scalar value) {
    return Log.FUNCTION.apply(MAX.apply(Sign.requirePositiveOrZero(value)));
  }
}
