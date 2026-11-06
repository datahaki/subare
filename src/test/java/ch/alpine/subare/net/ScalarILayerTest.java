// code by jph
package ch.alpine.subare.net;

import static org.junit.jupiter.api.Assertions.assertInstanceOf;

import java.util.concurrent.ThreadLocalRandom;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.num.Pi;

class ScalarILayerTest {
  @Test
  void testScalar() {
    int hidden = 10;
    NetLayer netLayer = NetChain.of( //
        new ScalarILayer(), //
        NetChains.sequence(() -> ElementwiseLayer.gelu(), 1, hidden, 1), //
        new ScalarOLayer() //
    );
    netLayer.initialize(ThreadLocalRandom.current());
    Tensor forward = netLayer.forward(RealScalar.ONE);
    assertInstanceOf(Scalar.class, forward);
    Tensor back = netLayer.back(Pi.VALUE);
    assertInstanceOf(Scalar.class, back);
    Tensor error = netLayer.error(Pi.VALUE);
    assertInstanceOf(Scalar.class, error);
  }

  @Test
  void testScalarL() {
    NetLayer netLayer = NetChain.of( //
        new ScalarILayer(), //
        new LinearLayer(1, 1), //
        new ScalarOLayer() //
    );
    netLayer.initialize(ThreadLocalRandom.current());
    Tensor forward = netLayer.forward(RealScalar.ONE);
    assertInstanceOf(Scalar.class, forward);
    Tensor back = netLayer.back(Pi.VALUE);
    assertInstanceOf(Scalar.class, back);
    Tensor error = netLayer.error(Pi.VALUE);
    assertInstanceOf(Scalar.class, error);
  }
}
