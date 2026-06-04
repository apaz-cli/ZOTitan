"""Unit tests for ObjectiveMixture composition — sampling, routing, weighting, eval.

Uses fake atomic objectives so no model/GPU/dataset is needed (the model path is
covered by tests/test_smoke.py)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter
from typing import Any

import pytest

from objective import (
    Score, ObjectiveMixture, MixtureComponent, make_objective,
    parse_objective_spec, build_objective, register_objective, OBJECTIVES,
)


class _FakeObjective:
    flat_batches   = True
    differentiable = True

    def __init__(self, name, val, sources=()):
        self.name = name
        self._val = val
        self._sources = list(sources)

    def dataset_sources(self):
        return self._sources

    def train_batches(self, tokenizer, seed, batch_size):
        i = 0
        while True:
            yield {"name": self.name, "i": i}
            i += 1

    def to_device(self, batch, device):
        return batch

    def score(self, model, batch):
        return Score(self._val, {"ce": self._val})

    def evaluate(self, model, tokenizer, n_examples=None, split=None):
        return {"metric": self._val}


def test_mixture_sampling_ratio():
    a, b = _FakeObjective("a", 1.0), _FakeObjective("b", 2.0)
    mix = ObjectiveMixture([MixtureComponent(a, data_weight=3.0),
                            MixtureComponent(b, data_weight=1.0)])
    it = mix.train_batches(None, seed=0, batch_size=1)
    counts = Counter(next(it)[0] for _ in range(4000))
    frac_a = counts[0] / 4000
    assert 0.70 < frac_a < 0.80, frac_a            # data_weight 3:1 → ≈0.75


def test_mixture_score_routes_and_weights():
    a, b = _FakeObjective("a", 1.0), _FakeObjective("b", 3.0)
    mix = ObjectiveMixture([MixtureComponent(a, loss_weight=1.0),
                            MixtureComponent(b, loss_weight=1.0)])   # normalized → 0.5 each
    s0 = mix.score(None, (0, {}))
    assert s0.value == 0.5 * 1.0
    assert s0.metrics["a/ce"] == 1.0
    assert s0.metrics["a/loss"] == 1.0             # the unweighted sub scalar
    s1 = mix.score(None, (1, {}))
    assert s1.value == 0.5 * 3.0


def test_mixture_single_component_unweighted():
    a = _FakeObjective("a", 1.5)
    mix = ObjectiveMixture([MixtureComponent(a)])   # loss_weight normalizes to 1.0
    assert mix.score(None, (0, {})).value == 1.5


def test_mixture_evaluate_merges():
    a, b = _FakeObjective("a", 1.0), _FakeObjective("b", 2.0)
    mix = ObjectiveMixture([MixtureComponent(a), MixtureComponent(b)])
    assert mix.evaluate(None, None) == {"a/metric": 1.0, "b/metric": 2.0}


def test_mixture_to_device_routes_tag():
    a = _FakeObjective("a", 1.0)
    mix = ObjectiveMixture([MixtureComponent(a)])
    assert mix.to_device((0, {"x": 1}), "cpu") == (0, {"x": 1})
    assert mix.flat_batches is False


def test_mixture_differentiable_propagates():
    a, b = _FakeObjective("a", 1.0), _FakeObjective("b", 2.0)
    b.differentiable = False
    mix = ObjectiveMixture([MixtureComponent(a), MixtureComponent(b)])
    assert mix.differentiable is False


def test_mixture_collects_all_dataset_sources():
    from objective import DatasetSource
    noop = lambda: None
    sa = DatasetSource("a", lambda: True, noop)
    sb = DatasetSource("b", lambda: True, noop)
    a = _FakeObjective("a", 1.0, sources=[sa])
    b = _FakeObjective("b", 2.0, sources=[sb])
    mix = ObjectiveMixture([MixtureComponent(a), MixtureComponent(b)])
    assert mix.dataset_sources() == [sa, sb]


def test_mixture_duplicate_names_disambiguated():
    a, b = _FakeObjective("scijudge", 1.0), _FakeObjective("scijudge", 2.0)
    mix = ObjectiveMixture([MixtureComponent(a), MixtureComponent(b)])
    assert mix.evaluate(None, None) == {"scijudge#0/metric": 1.0, "scijudge#1/metric": 2.0}
    assert mix.score(None, (0, {})).metrics["scijudge#0/ce"] == 1.0
    assert mix.score(None, (1, {})).metrics["scijudge#1/ce"] == 2.0


def test_factory_dispatch():
    sj = make_objective("scijudge")
    assert sj.name == "scijudge" and sj.differentiable and sj.flat_batches
    c4 = make_objective("c4")
    assert c4.name == "c4"


def test_register_objective_adds_to_registry():
    assert OBJECTIVES["scijudge"].__name__ == "SciJudgeObjective"

    @register_objective("toy")
    class _Toy:
        def __init__(self, compile_enabled=False, compile_mode=None, max_seq_len=None, **kwargs):
            self.kwargs = kwargs
    try:
        assert OBJECTIVES["toy"] is _Toy
        obj: Any = make_objective("toy", flavor="vanilla")
        assert obj.name == "toy" and obj.kwargs == {"flavor": "vanilla"}
        with pytest.raises(ValueError):           # double-register is rejected
            register_objective("toy")(_Toy)
    finally:
        OBJECTIVES.pop("toy", None)


def test_make_objective_unknown_name_raises():
    with pytest.raises(ValueError):
        make_objective("nope")


def test_make_objective_loss_kwargs():
    # loss settings are accepted as plain spec kwargs (land on the criterion as config)
    obj: Any = make_objective("scijudge", z_loss_weight=0.1, fused=False)
    crit = obj.criteria[0][0]
    assert crit.z_loss_weight == 0.1 and crit.fused is False


def test_make_objective_split_and_loss_kwargs():
    # scijudge peels off its own `split` knob; the rest pass through as loss kwargs
    obj: Any = make_objective("scijudge", split="test_ood_year", z_loss_weight=0.1, fused=False)
    assert obj._split == "test_ood_year"


def test_build_objective_with_split_and_loss_kwargs():
    # the full spec -> build path for a mixed split + loss kwargs component
    obj: Any = build_objective(
        parse_objective_spec("(scijudge,split='test_ood_year',z_loss_weight=0.1,fused=False)"))
    assert obj.name == "scijudge" and obj._split == "test_ood_year"


def test_make_objective_unknown_kwarg_raises():
    with pytest.raises(TypeError):
        make_objective("scijudge", bogus=1)


def test_parse_spec_bare_name():
    objs = parse_objective_spec("scijudge")
    assert len(objs) == 1
    assert objs[0].dataset == "scijudge"
    assert objs[0].loss_weight == 1.0 and objs[0].data_weight == 1.0


def test_parse_spec_mixture_with_defaults():
    # "(c4,.5)+scijudge" -> [(c4, lw=.5, dw=1), (scijudge, lw=1, dw=1)]
    objs = parse_objective_spec("(c4,.5)+scijudge")
    assert [o.dataset for o in objs] == ["c4", "scijudge"]
    assert (objs[0].loss_weight, objs[0].data_weight) == (0.5, 1.0)
    assert (objs[1].loss_weight, objs[1].data_weight) == (1.0, 1.0)


def test_parse_spec_full_tuple_and_whitespace():
    objs = parse_objective_spec(" (scijudge, 2, 3) + c4 ")
    assert (objs[0].loss_weight, objs[0].data_weight) == (2.0, 3.0)
    assert objs[1].dataset == "c4"


def test_parse_spec_kwargs_typed():
    objs = parse_objective_spec("(scijudge,rank=64,eps=1e-3,flag=True,x=None)")
    assert objs[0].kwargs == {"rank": 64, "eps": 1e-3, "flag": True, "x": None}
    assert isinstance(objs[0].kwargs["rank"], int)
    assert isinstance(objs[0].kwargs["eps"], float)


def test_parse_spec_kwargs_after_weights():
    objs = parse_objective_spec("(c4,.5,2,split='val')")
    assert (objs[0].loss_weight, objs[0].data_weight) == (0.5, 2.0)
    assert objs[0].kwargs == {"split": "val"}


def test_parse_spec_quoted_value_with_delimiters():
    # comma, paren, plus, equals inside a single-quoted value are all literal
    objs = parse_objective_spec("(scijudge,note='a,b+c=(d)')+c4")
    assert objs[0].kwargs == {"note": "a,b+c=(d)"}
    assert [o.dataset for o in objs] == ["scijudge", "c4"]


def test_parse_spec_doubled_quote_escape():
    objs = parse_objective_spec("(scijudge,note='it''s')")
    assert objs[0].kwargs == {"note": "it's"}


def test_parse_spec_double_quoted_value():
    # double quotes work too (pass as "\"thing\"" on the shell); delimiters inside are literal
    objs = parse_objective_spec('(scijudge,note="a,b+c=(d)")+c4')
    assert objs[0].kwargs == {"note": "a,b+c=(d)"}
    assert [o.dataset for o in objs] == ["scijudge", "c4"]


def test_parse_spec_double_quote_doubled_escape():
    objs = parse_objective_spec('(scijudge,note="say ""hi""")')
    assert objs[0].kwargs == {"note": 'say "hi"'}


def test_parse_spec_errors():
    with pytest.raises(ValueError):
        parse_objective_spec("scijudge+")        # empty component
    with pytest.raises(ValueError):
        parse_objective_spec("(c4,1,2,3)")       # too many positional fields
    with pytest.raises(ValueError):
        parse_objective_spec("(scijudge,id=some/repo)")  # bare string, must quote
    with pytest.raises(ValueError):
        parse_objective_spec("(c4,x=1,2)")       # positional after kwarg
    with pytest.raises(ValueError):
        parse_objective_spec("(scijudge,note='unterminated)")  # bad quote


def test_build_objective_single_vs_mixture():
    single = build_objective(parse_objective_spec("scijudge"))
    assert single.name == "scijudge" and not isinstance(single, ObjectiveMixture)
    mix = build_objective(parse_objective_spec("(c4,.5)+scijudge"))
    assert isinstance(mix, ObjectiveMixture)
    assert mix.name == "mix(c4,scijudge)"
