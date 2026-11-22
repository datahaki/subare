// code by jph
package ch.alpine.subare.net.lay;

import java.util.List;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLayers;
import ch.alpine.subare.net.TrainableNetLayer;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.chq.DeterminateTensorQ;

class ResidualLayerTest {
  @Test
  void testInitialize() {
    NetLayer ll = new LinearLayer(2, 2);
    ResidualLayer residualLayer = new ResidualLayer(ll);
    NetLayers.initialize(residualLayer);
    Tensor y = residualLayer.forward(Tensors.vector(10, 10));
    DeterminateTensorQ.require(y);
  }

  @Test
  void testNested() {
    NetLayer netLayer = NetChains.argMax( //
        new ResidualLayer(NetChains.linear(ElementwiseLayers.GELU, 2, 7, 3)));
    NetLayers.initialize(netLayer);
    IO.println(netLayer);
    List<TrainableNetLayer> list = NetLayers.trainableLayers(netLayer).toList();
    for (TrainableNetLayer nl : list) {
      IO.println(nl);
    }
  }
}
