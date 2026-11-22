// code by jph
package ch.alpine.subare.net.lay;

import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.net.NetLayers;
import ch.alpine.tensor.Tensors;

class SingleEmbeddingLayerTest {
  @Test
  void test() {
    ScalarEmbeddingLayer singleEmbeddingLayer = new ScalarEmbeddingLayer(3, 2);
    NetLayers.initialize(singleEmbeddingLayer);
    assertThrows(Exception.class, () -> singleEmbeddingLayer.forward(Tensors.vector(3)));
    singleEmbeddingLayer.forward(Tensors.vector(1));
    singleEmbeddingLayer.back(Tensors.vector(.1, .2));
  }

  @Test
  void testBatch() {
    ScalarEmbeddingLayer singleEmbeddingLayer = new ScalarEmbeddingLayer(3, 2);
    NetLayers.initialize(singleEmbeddingLayer);
    singleEmbeddingLayer.forwardBatch(Tensors.fromString("{{1},{0}}"));
    singleEmbeddingLayer.backBatch(Tensors.fromString("{{.2,.3},{.1,.4}}"));
  }
}
