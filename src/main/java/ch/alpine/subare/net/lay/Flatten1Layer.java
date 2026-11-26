// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;

import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.alg.TakeList;
import ch.alpine.tensor.ext.Integers;
import ch.alpine.tensor.io.MathematicaFormat;

/** @see TakeListLayer
 * 
 * <p>inspired by
 * <a href="https://reference.wolfram.com/language/ref/FlattenLayer.html">FlattenLayer</a> */
public class Flatten1Layer implements NetLayer, Serializable {
  private int[] lengths = null;

  @Override
  public Tensor forward(Tensor x) {
    update(x);
    return _forward(x);
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    update(X.get(0));
    return Slash.of(X, Flatten1Layer::_forward);
  }

  public static Tensor _forward(Tensor x) {
    return Flatten.of(x, 1);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return TakeList.of(gradOutput, lengths);
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    return Slash.of(gradOutput, this::back);
  }

  private void update(Tensor x) {
    lengths = x.stream() //
        .mapToInt(Tensor::length) //
        .map(Integers::requirePositiveOrZero) //
        .toArray();
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("Flatten1Layer", lengths == null ? null : Integers.asList(lengths));
  }
}
