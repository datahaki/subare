// code by jph
package ch.alpine.subare.net;

import java.io.Serializable;
import java.util.random.RandomGenerator;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;

/** input Scalar -> output Tensor of length 1 */
public class ScalarILayer implements NetLayer, Serializable {
  Tensor outputCache = null;

  @Override
  public void initialize(RandomGenerator randomGenerator) {
    outputCache = null;
  }

  @Override
  public Tensor forward(Tensor x) {
    return outputCache = Tensors.of(x); // scalar -> vector[1]
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return gradOutput.Get(0); // vector[1] -> scalar
  }

  @Override
  public Tensor error(Tensor y) {
    return y.subtract(outputCache);
  }
}
