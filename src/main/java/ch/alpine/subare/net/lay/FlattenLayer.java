// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;
import java.util.List;

import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.ArrayReshape;
import ch.alpine.tensor.alg.Dimensions;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.io.MathematicaFormat;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/FlattenLayer.html">FlattenLayer</a> */
public class FlattenLayer implements NetLayer, Serializable {
  private transient List<Integer> list;

  @Override
  public Tensor forward(Tensor x) {
    list = Dimensions.of(x);
    return Flatten.of(x);
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    list = Dimensions.of(X);
    return Slash.of(X, Flatten::of);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return ArrayReshape.of(gradOutput, list);
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    return ArrayReshape.of(gradOutput, list);
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("FlattenLayer");
  }
}
