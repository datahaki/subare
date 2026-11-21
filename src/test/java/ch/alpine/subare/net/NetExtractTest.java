package ch.alpine.subare.net;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.net.lay.LinearLayer;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Dimensions;

class NetExtractTest {
  @Test
  void test() {
    List<Tensor> list = NetExtract.parameters(new LinearLayer(3, 4));
    assertEquals(Dimensions.of(list.get(0)), List.of(3, 4));
    assertEquals(Dimensions.of(list.get(1)), List.of(4, 1));
  }
}
