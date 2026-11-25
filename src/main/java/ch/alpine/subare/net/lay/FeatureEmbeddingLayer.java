// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Unprotect;
import ch.alpine.tensor.alg.Transpose;
import ch.alpine.tensor.io.MathematicaFormat;

public class FeatureEmbeddingLayer implements NetLayer, Serializable {
  public static NetLayer of(List<NetLayer> list) {
    return new FeatureEmbeddingLayer(list);
  }

  public static NetLayer of(NetLayer... featureSlices) {
    return of(Arrays.asList(featureSlices));
  }

  // ---
  private final List<NetLayer> list;

  private FeatureEmbeddingLayer(List<NetLayer> list) {
    this.list = list;
  }

  @Override
  public Tensor forward(Tensor x) {
    int beg = 0;
    List<Tensor> outs = new ArrayList<>(list.size());
    for (NetLayer netLayer : list) {
      outs.add(netLayer.forward(x.get(beg)));
      ++beg;
    }
    return Unprotect.using(outs);
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    int beg = 0;
    List<Tensor> outs = new ArrayList<>(list.size());
    for (NetLayer netLayer : list) {
      outs.add(netLayer.forwardBatch(X.get(Tensor.ALL, beg)));
      ++beg;
    }
    return Transpose.of(Unprotect.using(outs));
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    Tensor[] array = gradOutput.stream().toArray(Tensor[]::new);
    int beg = 0;
    List<Tensor> gW = new ArrayList<>(list.size());
    for (NetLayer netLayer : list) {
      gW.add(netLayer.back(array[beg]));
      ++beg;
    }
    return Unprotect.using(gW);
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    List<Tensor> gW = new ArrayList<>(list.size());
    int beg = 0;
    for (NetLayer netLayer : list) {
      gW.add(netLayer.backBatch(gradOutput.get(Tensor.ALL, beg)));
      ++beg;
    }
    return Transpose.of(Unprotect.using(gW));
  }

  @Override
  public Stream<NetLayer> nested() {
    return list.stream();
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("FeatureEmbeddingLayer", list.size());
  }
}
