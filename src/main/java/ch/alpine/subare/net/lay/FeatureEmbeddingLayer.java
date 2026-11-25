// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Unprotect;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.ext.Integers;
import ch.alpine.tensor.io.MathematicaFormat;

public class FeatureEmbeddingLayer implements NetLayer, Serializable {
  public static NetLayer of(List<LayerSlice> list) {
    return new FeatureEmbeddingLayer(list);
  }

  public static NetLayer of(LayerSlice... featureSlices) {
    return of(Arrays.asList(featureSlices));
  }

  // ---
  private final List<LayerSlice> list;
  private final int all_in;
  private final int all_out;

  private FeatureEmbeddingLayer(List<LayerSlice> list) {
    this.list = list;
    all_in = list.stream().mapToInt(fS -> fS.d_in()).sum();
    all_out = list.stream().mapToInt(fS -> fS.d_out()).sum();
  }

  @Override
  public Tensor forward(Tensor x) {
    int beg = 0;
    List<Tensor> outs = new ArrayList<>(list.size());
    for (LayerSlice layerSlice : list) {
      Tensor y = layerSlice.netLayer().forward(x.block(List.of(beg), List.of(layerSlice.d_in())));
      VectorQ.requireLength(y, layerSlice.d_out());
      outs.add(y);
      beg += layerSlice.d_in();
    }
    Integers.requireEquals(beg, all_in);
    return Unprotect.using(outs);
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    int batch_size = X.length();
    int beg = 0;
    List<Tensor> outs = new ArrayList<>(list.size());
    for (LayerSlice layerSlice : list) {
      outs.add(layerSlice.netLayer().forwardBatch(X.block(List.of(0, beg), List.of(batch_size, layerSlice.d_in()))));
      beg += layerSlice.d_in();
    }
    Integers.requireEquals(beg, all_in);
    return Tensor.of(IntStream.range(0, batch_size) //
        .mapToObj(i -> Tensor.of(outs.stream().map(o -> o.get(i)), outs.size())), //
        batch_size);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    Tensor[] array = gradOutput.stream().toArray(Tensor[]::new);
    List<Tensor> gW = new ArrayList<>(list.size());
    int beg = 0;
    for (LayerSlice layerSlice : list) {
      gW.add(layerSlice.netLayer().back(array[beg]));
      ++beg;
    }
    return null;
    // Unprotect.using(gW);
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    int batch_size = gradOutput.length();
    List<Tensor> gW = new ArrayList<>(list.size());
    int beg = 0;
    for (LayerSlice layerSlice : list) {
      gW.add(layerSlice.netLayer().backBatch(gradOutput.get(Tensor.ALL, beg)));
      ++beg;
    }
    // Integers.requireEquals(beg, all_out);
    return null;
    // Tensor.of(IntStream.range(0, batch_size) //
    // .mapToObj(i -> Tensor.of(gW.stream().map(o -> o.get(i)), gW.size())), //
    // batch_size);
  }

  @Override
  public Stream<NetLayer> nested() {
    return list.stream().map(LayerSlice::netLayer);
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("FeatureEmbeddingLayer", all_in, all_out);
  }
}
