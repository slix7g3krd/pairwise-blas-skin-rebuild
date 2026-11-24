import pytest
import graphics.blas_skin_rebuild as mod

CASES = [
    ("queue_fence", (1,)),
    ("instance_generation", (1,)),
    ("defer_release", (1,)),
    ("xform_bind_version", (1,)),
    ("merge_frame_builds", (1,)),
    ("validate_instance", (1,)),
    ("build_blas", (1,)),
    ("build_tlas", (1,)),
    ("trace", (1,)),
    ("release", (1,)),
    ("validate_rt", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
