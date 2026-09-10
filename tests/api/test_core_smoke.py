"""Minimal public-API path used by the installed-wheel CI smoke."""

import pandas as pd
import torch

from kpnn2 import (
    LayeredSpec,
    MaskedLinear,
    align_inputs,
    gather_hop_inputs,
    map_node_attributions,
    parse_layered,
)


def test_core_smoke():
    edgelist = pd.DataFrame(
        {
            "source": ["feature_a", "feature_b", "hidden"],
            "target": ["hidden", "hidden", "prediction"],
        }
    )
    spec = parse_layered(edgelist)
    assert isinstance(spec, LayeredSpec)
    assert spec.input_nodes == ("feature_a", "feature_b")

    data = pd.DataFrame(
        {
            "feature_b": [1.0, 2.0],
            "feature_a": [3.0, 4.0],
        }
    )
    x = align_inputs(data, spec)
    assert x.shape == (2, 2)
    assert x.dtype == torch.float32

    hop = spec.hops[0]
    assert hop.source_layers == (0,)
    layer = MaskedLinear(hop.to_mask())
    h = layer(
        gather_hop_inputs(
            {0: x},
            hop,
        )
    )
    assert h.shape == (2, 1)

    da = map_node_attributions(
        attributions=h,
        spec=spec,
        layer=1,
    )
    assert list(da["node"].values) == list(spec.layer_nodes[1])
    assert int(da.sizes["observation"]) == 2
    assert int(da.coords["layer"]) == 1


if __name__ == "__main__":
    test_core_smoke()
    print("installed wheel smoke test passed")
