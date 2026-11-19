// code by jph
package ch.alpine.subare.net.lay;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.net.NetInitialize;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Dimensions;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.d.DiscreteUniformDistribution;

class EmbeddingLayerTest {
  @Test
  void test() {
    EmbeddingLayer embeddingLayer = new EmbeddingLayer(3, 4);
    NetInitialize.of(embeddingLayer);
    Tensor y = embeddingLayer.forward(Tensors.vector(0, 1, 2, 1, 0));
    assertEquals(Dimensions.of(y), List.of(5 * 4));
    embeddingLayer.back(y);
  }

  @Test
  void testZero() {
    EmbeddingLayer embeddingLayer = new EmbeddingLayer(3, 4);
    NetInitialize.of(embeddingLayer);
    Tensor y = embeddingLayer.forward(Tensors.vector(0, 2, 2, 2, 0));
    assertEquals(Dimensions.of(y), List.of(5 * 4));
    Tensor back = embeddingLayer.back(y);
    // assertEquals(back.get(1), Array.zeros(4));
  }

  @Test
  void testBatch() {
    EmbeddingLayer embeddingLayer = new EmbeddingLayer(3, 4);
    NetInitialize.of(embeddingLayer);
    Tensor X = RandomVariate.of(DiscreteUniformDistribution.forArray(3), 10, 5);
    Tensor Y = embeddingLayer.forwardBatch(X);
    assertEquals(Dimensions.of(Y), List.of(10, 5 * 4));
    Tensor back = embeddingLayer.backBatch(Y);
    // assertEquals(back.get(1), Array.zeros(4));
    // IO.println(Dimensions.of(back));
  }
}
