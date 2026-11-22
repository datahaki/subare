// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

import ch.alpine.subare.net.NetLayer;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Join;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.io.MathematicaFormat;

public class FeatureEmbeddingLayer implements NetLayer, Serializable {
  public static record NetFeature(int d_in, int d_out, NetLayer netLayer) implements Serializable {
  }

  public static NetLayer of(NetFeature... netFeatures) {
    return new FeatureEmbeddingLayer(Arrays.asList(netFeatures));
  }

  private final List<NetFeature> list;
  private transient List<Tensor> gW;

  public FeatureEmbeddingLayer(List<NetFeature> list) {
    this.list = list;
  }

  @Override
  public Tensor forward(Tensor x) {
    int beg = 0;
    List<Tensor> outs = new ArrayList<>(list.size());
    for (NetFeature netStack : list) {
      Tensor y = netStack.netLayer.forward(x.block(List.of(beg), List.of(netStack.d_in)));
      VectorQ.requireLength(y, netStack.d_out);
      outs.add(y);
      beg += netStack.d_in;
    }
    return Tensor.of(outs.stream().flatMap(Tensor::stream));
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    int beg = 0;
    List<Tensor> outs = new ArrayList<>(list.size());
    for (NetFeature netStack : list) {
      outs.add(netStack.netLayer.forwardBatch(X.block(List.of(0, beg), List.of(X.length(), netStack.d_in))));
      beg += netStack.d_in;
    }
    return Join.of(1, outs);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    gW = new ArrayList<>(list.size());
    int beg = 0;
    for (NetFeature netStack : list) {
      Tensor g = gradOutput.block(List.of(beg), List.of(netStack.d_out));
      gW.add(netStack.netLayer().back(g));
    }
    return null;
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    gW = new ArrayList<>(list.size());
    int beg = 0;
    for (NetFeature netStack : list) {
      Tensor G = gradOutput.block(List.of(0, beg), List.of(gradOutput.length(), netStack.d_out));
      gW.add(netStack.netLayer().backBatch(G));
    }
    return null;
  }

  @Override
  public Stream<NetLayer> nested() {
    return list.stream().map(NetFeature::netLayer);
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("FeatureEmbeddingLayer", list.size());
  }
}
