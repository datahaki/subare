// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

import ch.alpine.subare.net.NetLayer;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Join;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.ext.Integers;
import ch.alpine.tensor.io.MathematicaFormat;

public class FeatureEmbeddingLayer implements NetLayer, Serializable {
  public static record FeatureSlice(int d_in, int d_out, NetLayer netLayer) implements Serializable {
    public FeatureSlice {
      Integers.requirePositive(d_in);
      Integers.requirePositive(d_out);
      Objects.requireNonNull(netLayer);
    }
  }

  public static NetLayer of(List<FeatureSlice> list) {
    return new FeatureEmbeddingLayer(list);
  }

  public static NetLayer of(FeatureSlice... featureSlices) {
    return of(Arrays.asList(featureSlices));
  }

  private final List<FeatureSlice> list;
  private transient List<Tensor> gW;

  private FeatureEmbeddingLayer(List<FeatureSlice> list) {
    this.list = list;
  }

  @Override
  public Tensor forward(Tensor x) {
    int beg = 0;
    List<Tensor> outs = new ArrayList<>(list.size());
    for (FeatureSlice featureSlice : list) {
      Tensor y = featureSlice.netLayer.forward(x.block(List.of(beg), List.of(featureSlice.d_in)));
      VectorQ.requireLength(y, featureSlice.d_out);
      outs.add(y);
      beg += featureSlice.d_in;
    }
    return Tensor.of(outs.stream().flatMap(Tensor::stream));
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    int beg = 0;
    List<Tensor> outs = new ArrayList<>(list.size());
    for (FeatureSlice featureSlice : list) {
      outs.add(featureSlice.netLayer.forwardBatch(X.block(List.of(0, beg), List.of(X.length(), featureSlice.d_in))));
      beg += featureSlice.d_in;
    }
    return Join.of(1, outs);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    gW = new ArrayList<>(list.size());
    int beg = 0;
    for (FeatureSlice featureSlice : list) {
      Tensor g = gradOutput.block(List.of(beg), List.of(featureSlice.d_out));
      gW.add(featureSlice.netLayer().back(g));
      beg += featureSlice.d_out;
    }
    return null;
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    gW = new ArrayList<>(list.size());
    int beg = 0;
    for (FeatureSlice featureSlice : list) {
      Tensor G = gradOutput.block(List.of(0, beg), List.of(gradOutput.length(), featureSlice.d_out));
      gW.add(featureSlice.netLayer().backBatch(G));
      beg += featureSlice.d_out;
    }
    return null;
  }

  @Override
  public Stream<NetLayer> nested() {
    return list.stream().map(FeatureSlice::netLayer);
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("FeatureEmbeddingLayer", list.size());
  }
}
