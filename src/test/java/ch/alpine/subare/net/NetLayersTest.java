// code by jph
package ch.alpine.subare.net;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.LinkedList;
import java.util.List;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.mat.MatrixQ;

class NetLayersTest {
  @Test
  void testScalar() {
    List<Tensor> list = new LinkedList<>();
    assertThrows(Exception.class, () -> NetLayers.build(list, Tensors.fromString("1")));
  }

  @Test
  void testVector() {
    List<Tensor> list = new LinkedList<>();
    NetLayers.build(list, Tensors.fromString("{1}"));
    assertEquals(list.size(), 1);
    MatrixQ.require(list.get(0));
  }
}
