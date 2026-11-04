// code by jph
package ch.alpine.subare.net;

import java.io.Serializable;
import java.util.random.RandomGenerator;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.ext.Integers;

/** input Tensor of length 1 -> output Scalar */
public class ScalarOLayer implements NetLayer, Serializable {
  Scalar outputCache;

  @Override
  public void initialize(RandomGenerator randomGenerator) {
    outputCache = null;
  }

  @Override
  public Scalar forward(Tensor x) {
    Integers.requireEquals(x.length(), 1);
    return outputCache = x.Get(0);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return Tensors.of(gradOutput);
  }

  @Override
  public Tensor error(Tensor y) {
    return y.subtract(outputCache);
  }
}
