// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;
import java.util.stream.IntStream;

import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.alg.TakeList;
import ch.alpine.tensor.ext.Integers;
import ch.alpine.tensor.io.MathematicaFormat;

/** @see Flatten1Layer */
public class TakeListLayer implements NetLayer, Serializable {
  private final int[] lengths;

  public TakeListLayer(int... lengths) {
    this.lengths = lengths;
    IntStream.of(lengths).forEach(Integers::requirePositiveOrZero);
  }

  @Override
  public Tensor forward(Tensor x) {
    return TakeList.of(x, lengths);
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    return Slash.of(X, this::forward);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return Flatten.of(gradOutput, 1);
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    return Slash.of(gradOutput, this::back);
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("TakeListLayer", Integers.asList(lengths));
  }
}
