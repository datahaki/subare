// code by jph
package ch.alpine.subare.net.mod;

import java.util.ArrayList;
import java.util.List;

import ch.alpine.subare.net.lay.FeatureEmbeddingLayer;
import ch.alpine.subare.net.lay.LayerSlice;
import ch.alpine.subare.net.los.CombinedLoss;

public enum SplitModel {
  ;
  public static NetModel of(List<ModelSlice> list) {
    List<LayerSlice> layers = new ArrayList<>();
    for (ModelSlice modelSlice : list) {
      layers.add(new LayerSlice(modelSlice.d_in(), modelSlice.d_out(), modelSlice.netModel().netLayer()));
    }
    return new NetModel(FeatureEmbeddingLayer.of(layers), new CombinedLoss(null));
  }
}
