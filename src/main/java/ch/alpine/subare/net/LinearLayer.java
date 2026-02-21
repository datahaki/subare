// code by jph
package ch.alpine.subare.net;

import java.util.random.RandomGenerator;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.lie.TensorProduct;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/LinearLayer.html">LinearLayer</a> */
public class LinearLayer implements Layer {
  public static LinearLayer of(Distribution d, RandomGenerator randomGenerator, int ante, int post) {
    LinearLayer linearLayer = new LinearLayer();
    linearLayer.W = RandomVariate.of(d, randomGenerator, ante, post);
    linearLayer.b = Array.zeros(ante);
    return linearLayer;
  }

  Tensor W;
  Tensor b;
  /** input cache */
  Tensor inputCache;
  Tensor gW;
  Tensor gb;

  @Override
  public Tensor forward(Tensor x) {
    this.inputCache = x;
    return W.dot(x).add(b);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    // IO.println("L L recv " + gradOutput.maps(Round._3));
    gW = TensorProduct.of(gradOutput, inputCache);
    gb = gradOutput;
    return gradOutput.dot(W); // gradInput
  }

  @Override
  public void update() {
    W = W.add(gW);
    b = b.add(gb);
  }

  @Override
  public Tensor error(Tensor y) {
    throw new IllegalStateException();
  }

  @Override
  public Tensor parameters() {
    return Flatten.of(W, b);
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("LinearLayer", W, b);
  }
}
