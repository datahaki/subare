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
import ch.alpine.tensor.ext.Int;
import ch.alpine.tensor.ext.Integers;
import ch.alpine.tensor.red.Mean;

/** Loss == Cross Entropy divided by batch_size */
public record CrossEntropyBatchLoss(Scalar loss, Tensor gradOutput) implements NetLoss, Serializable {
  public static NetLoss of(Tensor interCache, Tensor target) {
    Integers.requireEquals(interCache.length(), target.length());
    int batch_size = target.length();
    int[] index = target.stream() //
        .map(Scalar.class::cast) //
        .mapToInt(Scalars::intValueExact) //
        .toArray();
    Scalar loss = (Scalar) Mean.of(IntStream.range(0, batch_size) //
        .mapToObj(i -> interCache.Get(i, index[i])) //
        .map(StaticHelper::safeLog)).negate();
    Tensor error = interCache.copy();
    Int myI = new Int();
    error.stream() //
        .forEach(row -> row.set(RealScalar.ONE.negate()::add, index[myI.getAndIncrement()]));
    Tensor gradOutput = error.multiply(Rational.of(1, batch_size));
    return new CrossEntropyBatchLoss(loss, gradOutput);
  }
}
