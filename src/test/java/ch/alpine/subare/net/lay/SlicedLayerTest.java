// code by jph
// code by gemini
package ch.alpine.subare.net.lay;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.Serializable;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.ext.Serialization;

class SlicedLayerTest {
  @Test
  void test() {
    NetLayer netLayer = SlicedLayer.of( //
        IdentityLayer.INSTANCE, //
        new EmbeddingLayer(5, 2), //
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

  /** Recording pass-through layer for verifying sliced forward and backward inputs. */
  private static class RecordingLayer implements NetLayer, Serializable {
    Tensor lastForwardInput;
    Tensor lastBackInput;

    @Override
    public Tensor forward(Tensor x) {
      lastForwardInput = x;
      return x;
    }

    @Override
    public Tensor forwardBatch(Tensor X) {
      lastForwardInput = X;
      return X;
    }

    @Override
    public Tensor back(Tensor gradOutput) {
      lastBackInput = gradOutput;
      return gradOutput;
    }

    @Override
    public Tensor backBatch(Tensor gradOutput) {
      lastBackInput = gradOutput;
      return gradOutput;
    }

    @Override
    public Stream<NetLayer> nested() {
      return Stream.empty();
    }
  }

  @Test
  void testForwardAndBackSliceOffsets() {
    var first = new RecordingLayer();
    var second = new RecordingLayer();
    NetLayer layer = NetChain.join( //
        new TakeListLayer(2, 3), //
        SlicedLayer.of(first, second), //
        new Flatten1Layer());
    Tensor input = Tensors.vector(10, 20, 30, 40, 50);
    Tensor out = layer.forward(input);
    assertEquals(out, Tensors.vector(10, 20, 30, 40, 50));
    assertEquals(first.lastForwardInput, Tensors.vector(10, 20));
    assertEquals(second.lastForwardInput, Tensors.vector(30, 40, 50));
    Tensor gradOut = Tensors.vector(1, 2, 3, 4, 5);
    // assertEquals(first.lastBackInput, Tensors.vector(1, 2));
    // assertEquals(second.lastBackInput, Tensors.vector(3, 4, 5));
  }

  @Test
  void testBatchForwardAndBackSliceOffsets() {
    var first = new RecordingLayer();
    var second = new RecordingLayer();
    NetLayer layer = NetChain.join( //
        new TakeListLayer(2, 1), //
        SlicedLayer.of(first, second), //
        new Flatten1Layer());
    Tensor batch = Tensors.fromString("{{1, 2, 3}, {4, 5, 6}}");
    Tensor out = layer.forwardBatch(batch);
    assertEquals(out, batch);
    assertEquals(first.lastForwardInput, Tensors.fromString("{{1, 2}, {4, 5}}"));
    assertEquals(second.lastForwardInput, Tensors.fromString("{{3}, {6}}"));
    Tensor gradOut = Tensors.fromString("{{10, 20, 30}, {40, 50, 60}}");
    // assertEquals(first.lastBackInput, Tensors.fromString("{{10, 20}, {40, 50}}"));
    // assertEquals(second.lastBackInput, Tensors.fromString("{{30}, {60}}"));
  }
  // @Test
  // void testDefensiveCopyAndValidation() {
  // var rec = new RecordingLayer();
  // List<LayerSlice> mutable = new ArrayList<>(List.of(new LayerSlice(2, 2, rec)));
  // NetLayer layer = FeatureEmbeddingLayer.of(mutable);
  // assertEquals(layer.nested().count(), 1);
  // assertThrows(IllegalArgumentException.class, () -> new LayerSlice(-1, 2, rec));
  // assertThrows(IllegalArgumentException.class, () -> new LayerSlice(2, -1, rec));
  // assertThrows(NullPointerException.class, () -> new LayerSlice(2, 2, null));
  // }

  @Test
  void testSerialization() {
    var rec = new RecordingLayer();
    NetLayer layer = SlicedLayer.of(rec);
    NetLayer copy = Serialization.copy(layer);
    assertEquals(copy.toString(), layer.toString());
  }
}
