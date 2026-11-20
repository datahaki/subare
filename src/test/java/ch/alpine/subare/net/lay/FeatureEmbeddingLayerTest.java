package ch.alpine.subare.net.lay;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.net.NetInitialize;
import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.subare.net.lay.FeatureEmbeddingLayer.NetFeature;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.mat.MatrixQ;

class FeatureEmbeddingLayerTest {
  @Test
  void test() {
    NetLayer netLayer = FeatureEmbeddingLayer.of( //
        new NetFeature(1, 1, IdentityLayer.INSTANCE), //
        new NetFeature(1, 2, new ScalarEmbeddingLayer(5, 2)), //
        new NetFeature(2, 2, IdentityLayer.INSTANCE));
    NetInitialize.of(netLayer);
    {
      Tensor y = netLayer.forward(Tensors.vector(0.2, 4, 0.1, 0.3));
      VectorQ.requireLength(y, 5);
      netLayer.back(y);
    }
    {
      Tensor Y = netLayer.forwardBatch(Tensors.of( //
          Tensors.vector(0.2, 4, 0.1, 0.3), //
          Tensors.vector(0.1, 0, 0.4, 1.3)) //
      );
      MatrixQ.requireSize(Y, 2, 5);
      netLayer.backBatch(Y);
    }
  }
}
