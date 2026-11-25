// code by jph
package ch.alpine.subare.net.lay;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Range;

class Flatten1LayerTest {
  @Test
  void testForward() {
    Flatten1Layer join0Layer = new Flatten1Layer();
    Tensor x = Tensors.fromString("{{1},{2,3}}");
    Tensor forward = join0Layer.forward(x);
    assertEquals(forward, Range.of(1, 4));
    Tensor back = join0Layer.back(forward);
    assertEquals(back, x);
  }

  @Test
  void testForwardBatch() {
    Flatten1Layer join0Layer = new Flatten1Layer();
    Tensor x = Tensors.fromString("{{{1},{2,3}},{{4},{5,6}}}");
    Tensor forward = join0Layer.forwardBatch(x);
    Tensor expect = Tensors.fromString("{{1, 2, 3}, {4, 5, 6}}");
    assertEquals(forward, expect);
    Tensor back = join0Layer.backBatch(forward);
    assertEquals(back, x);
  }
}
