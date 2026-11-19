package ch.alpine.subare.net.lay;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.net.NetInitialize;
import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.chq.DeterminateTensorQ;

class ResidualLayerTest {
  @Test
  void testInitialize() {
    NetLayer ll = new LinearLayer(2, 2);
    ResidualLayer residualLayer = new ResidualLayer(ll);
    NetInitialize.of(residualLayer);
    Tensor y = residualLayer.forward(Tensors.vector(10, 10));
    DeterminateTensorQ.require(y);
  }
}
