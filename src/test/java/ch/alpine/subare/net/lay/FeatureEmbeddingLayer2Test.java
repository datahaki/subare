package ch.alpine.subare.net.lay;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;

class FeatureEmbeddingLayer2Test {
  @Test
  void test() {
    NetLayer embed = FeatureEmbeddingLayer.of( //
        IdentityLayer.INSTANCE, //
        new ScalarEmbeddingLayer(5, 2), //
        IdentityLayer.INSTANCE);
    NetLayer netLayer = NetChain.join(new TakeListLayer(1, 1, 2), embed);
    NetLayers.initialize(netLayer);
    {
      Tensor y = netLayer.forward(Tensors.vector(0.2, 4, 0.1, 0.3));
      assertEquals(y.length(), 3);
      Tensor z = netLayer.back(y);
    }
    {
      Tensor X = Tensors.of( //
          Tensors.vector(0.2, 4, 0.1, 0.3), //
          Tensors.vector(0.2, 4, 0.1, 0.3), //
          Tensors.vector(0.2, 4, 0.1, 0.3), //
          Tensors.vector(0.2, 4, 0.1, 0.3), //
          Tensors.vector(0.2, 4, 0.1, 0.3), //
          Tensors.vector(0.2, 4, 0.1, 0.3), //
          Tensors.vector(0.1, 0, 0.4, 1.3));
      Tensor Y = netLayer.forwardBatch(X);
      Tensor g = netLayer.backBatch(Y);
    }
  }
}
