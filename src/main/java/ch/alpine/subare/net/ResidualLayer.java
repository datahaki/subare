// code by jph
package ch.alpine.subare.net;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.io.MathematicaFormat;

public class ResidualLayer implements NetLayer {
  private final NetLayer netLayer;
  private Tensor outputCache;

  public ResidualLayer(NetLayer netLayer) {
    this.netLayer = netLayer;
  }

  @Override
  public Tensor forward(Tensor x) {
    return outputCache = netLayer.forward(x).add(x);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return netLayer.back(gradOutput).add(gradOutput);
  }

  @Override
  public void update() {
    netLayer.update();
  }

  @Override
  public Tensor error(Tensor y) {
    return y.subtract(outputCache);
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("ResidualLayer", netLayer);
  }
}
