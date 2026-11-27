package ch.alpine.subare.net.lay;

import java.io.Serializable;

import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.ConstantArray;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.lie.rot.AngleVector;

public record EnhanceLayer(Tensor matrix) implements NetLayer, Serializable {
  @Override
  public Tensor forward(Tensor x) {
    return Flatten.of(matrix.dot(x).maps(AngleVector::of));
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    return Slash.of(X, this::forward);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return Tensors.unmodifiableEmpty();
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    return ConstantArray.of(Tensors.empty(), gradOutput.length());
  }

  @Override
  public final String toString() {
    return MathematicaFormat.concise("Enhance", matrix);
  }
}
