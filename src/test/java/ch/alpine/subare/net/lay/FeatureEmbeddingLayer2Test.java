package ch.alpine.subare.net.lay;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;

class FeatureEmbeddingLayer2Test {
  @Test
  void test() {
    NetLayer netLayer = FeatureEmbeddingLayer.of( //
        IdentityLayer.INSTANCE, //
        new ScalarEmbeddingLayer(5, 2), //
        IdentityLayer.INSTANCE);
    NetLayers.initialize(netLayer);
    Tensor x = Tensors.fromString("{0.2, 4, {0.1, 0.3}}");
    {
      Tensor y = netLayer.forward(x);
      assertEquals(y.length(), 3);
      Tensor z = netLayer.back(y);
    }
    Tensor x2 = Tensors.fromString("{0.1, 4, {0.1, 0.3}}");
    {
      Tensor X = Tensors.of( //
          x, //
          x, //
          x, //
          x, //
          x, //
          x, //
          x2);
      Tensor Y = netLayer.forwardBatch(X);
      Tensor g = netLayer.backBatch(Y);
    }
  }
}
