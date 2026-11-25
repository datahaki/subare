package ch.alpine.subare.net.lay;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;

class TakeListLayerTest {
  @Test
  void test() {
    Distribution d = NormalDistribution.standard();
    TakeListLayer takeListLayer = new TakeListLayer(2, 3);
    NetLayer netLayer = NetChain.join(takeListLayer, new Flatten1Layer());
    {
      Tensor x = RandomVariate.of(d, 5);
      assertEquals(x, netLayer.forward(x));
      assertEquals(x, netLayer.back(x));
    }
    {
      Tensor x = RandomVariate.of(d, 10, 5);
      assertEquals(x, netLayer.forwardBatch(x));
      assertEquals(x, netLayer.backBatch(x));
    }
  }
}
