package ch.alpine.subare.net;

import ch.alpine.tensor.Tensor;

public class ResidualLayer implements Layer {
  private final NetChain netChain;

  public ResidualLayer(NetChain netChain) {
    this.netChain = netChain;
  }

  @Override
  public Tensor forward(Tensor x) {
    return netChain.forward(x).add(x);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return netChain.back(gradOutput).add(gradOutput);
  }

  @Override
  public void update() {
    netChain.update();
  }

  @Override
  public Tensor error(Tensor y) {
    return netChain.error(y);
  }
}
