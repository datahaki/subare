// code by jph
package ch.alpine.subare.net;

import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.ext.Serialization;

class NetChainTest {
  @Test
  void testSerial() throws ClassNotFoundException, IOException {
    NetLayer netLayer = NetChain.of( //
        new ResidualLayer(NetChains.linear(() -> ElementwiseLayer.gelu(), 3, 10, 3)), //
        new SoftArgMax());
    netLayer.initialize(ThreadLocalRandom.current());
    Serialization.copy(netLayer);
  }
}
