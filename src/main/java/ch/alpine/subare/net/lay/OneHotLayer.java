// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;

import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.mat.IdentityMatrix;

/** Embeds integers between 0 and (v-1) into a continuous vector space of dimension d.
 * 
 * Backpropagation stops at the embedding layer!
 * 
 * inspired by
 * <a href="https://reference.wolfram.com/language/ref/EmbeddingLayer.html">EmbeddingLayer</a> */
public class OneHotLayer implements NetLayer, Serializable {
  private final Tensor[] id;

  public OneHotLayer(int v) {
    id = IdentityMatrix.stream(v) //
        .map(Tensor::unmodifiable) //
        .toArray(Tensor[]::new);
  }

  @Override
  public Tensor forward(Tensor x) {
    VectorQ.requireLength(x, 1);
    // returns dimensions: [x.length() * d]
    return id[Scalars.intValueExact(x.Get(0))];
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
    return Array.zeros(gradOutput.length(), 0);
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("OneHotLayer", id.length);
  }
}
