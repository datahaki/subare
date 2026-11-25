package ch.alpine.subare.net.lay;

import java.io.Serializable;

import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Join;
import ch.alpine.tensor.alg.TakeList;
import ch.alpine.tensor.ext.Integers;
import ch.alpine.tensor.io.MathematicaFormat;

public class Join0Layer implements NetLayer, Serializable {
  private int[] lengths = null;

  @Override
  public Tensor forward(Tensor x) {
    update(x);
    return Join.of(0, x);
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    update(X.get(0));
    return Join.of(1, X);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return TakeList.of(gradOutput, lengths);
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    return Slash.of(gradOutput, s -> TakeList.of(s, lengths));
  }

  private void update(Tensor x) {
    lengths = x.stream() //
        .mapToInt(Tensor::length) //
        .map(Integers::requirePositiveOrZero) //
        .toArray();
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("Join0Layer");
  }
}
