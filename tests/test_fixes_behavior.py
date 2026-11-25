"""六个修复点的行为测试：证明缺陷场景被真正修掉，而非桩函数。"""

import pytest

from graphics.blas_skin_rebuild import (
    RTContext,
    FenceNotWaitedError,
    UseAfterFreeError,
    VersionMismatchError,
    queue_fence,
    instance_generation,
    defer_release,
    release,
    xform_bind_version,
    merge_frame_builds,
    validate_instance,
    build_blas,
    build_tlas,
    trace,
    validate_rt,
    fallback,
    determinism,
)


# --- 修复 1：跨队列插栅栏 -----------------------------------------------


def test_graphics_must_wait_compute_fence_before_tlas():
    ctx = RTContext()
    build_blas(ctx, 0, frame=1)
    # compute signal 到 1，graphics 时间线仍为 0：超前 wait 必须报错。
    with pytest.raises(FenceNotWaitedError):
        ctx.graphics_wait(99)
    tlas = build_tlas(ctx, frame=1)
    # 正常路径：build_tlas 内部完成 wait，TLAS 收录到实例。
    assert tlas.fence_value == ctx.graphics_timeline
    instance_generation(ctx, 0, 0, 1)
    tlas = build_tlas(ctx, frame=2)
    assert len(tlas.entries) == 1
    assert ctx.fences[tlas.fence_value].waited_by_graphics


# --- 修复 2：实例按代际标记 ---------------------------------------------


def test_slot_reuse_invalidates_old_generation():
    ctx = RTContext()
    build_blas(ctx, 0, frame=1)
    first = instance_generation(ctx, 0, 0, 1)
    old_handle = (first["slot"], first["generation"])

    # 角色销毁，槽位被新角色复用，代际递增。
    instance_generation(ctx, 0, 0, 1, destroy=True)
    second = instance_generation(ctx, 0, 0, 1)
    assert second["generation"] == first["generation"] + 2

    # 旧句柄校验失败；新句柄有效。
    assert validate_instance(ctx, *old_handle)["valid"] is False
    assert validate_instance(ctx, 0, second["generation"])["valid"] is True
    # 被销毁的实例也不能通过校验（同时无绑定、标记销毁）。
    instance_generation(ctx, 0, 0, 1, destroy=True)
    destroyed_reasons = validate_instance(ctx, 0)["reasons"]
    assert "instance_destroyed" in destroyed_reasons
    assert destroyed_reasons != []


# --- 修复 3：延迟释放旧结构 ---------------------------------------------


def test_old_blas_deferred_not_freed_until_fence_and_frame_pass():
    ctx = RTContext()
    v1 = build_blas(ctx, 7, frame=1)
    v2 = build_blas(ctx, 7, frame=2)  # v1 进入 pending_free，v2 ready
    assert v1.state == "pending_free"
    assert v2.state == "ready"

    # 同帧 release：帧未跨过，资源仍挂着（不能立即回收）。
    same = release(ctx, frame=2)
    assert same["freed_count"] == 0 and same["still_pending"] == 1

    # graphics 尚未 wait v2 的栅栏 => 仍不能释放。
    ctx.graphics_wait(v1.ready_fence)
    blocked = release(ctx, frame=3)
    assert blocked["freed_count"] == 0

    # 栅栏跨过 + 进入更晚帧后才真正 free。
    ctx.graphics_wait(v2.ready_fence)
    done = release(ctx, frame=3)
    assert done["freed_count"] == 1
    assert v1.state == "freed"
    assert v2.state == "ready"


def test_trace_never_touches_non_ready_blas():
    ctx = RTContext()
    build_blas(ctx, 3, frame=1)
    instance_generation(ctx, 0, 3, 1)
    build_tlas(ctx, frame=1)
    # 人为把 BLAS 置为非就绪态模拟回收中：查询护栏必须拦截。
    ctx.blas_current[3].state = "freed"
    with pytest.raises(UseAfterFreeError):
        trace(ctx, [{"slot": 0, "generation": 1,
                     "origin": (0, 0, 5), "direction": (0, 0, -1)}])


# --- 修复 4：变换与网格版本绑定 -----------------------------------------


def test_xform_binding_is_versioned_and_immutable():
    ctx = RTContext()
    binding = xform_bind_version(ctx, 3, translate=(1.0, 2.0, 3.0))
    assert binding.mesh_version == 3
    with pytest.raises(AttributeError):
        binding.mesh_version = 4  # 不可变：主/阴影光线共用同一份绑定
    with pytest.raises(VersionMismatchError):
        xform_bind_version(ctx, 3, expected_version=2)


def test_instance_binding_rejects_stale_mesh_version():
    ctx = RTContext()
    build_blas(ctx, 5, frame=1, mesh_version=1)
    v2 = build_blas(ctx, 5, frame=2, mesh_version=2)
    assert v2.version == 2
    with pytest.raises(VersionMismatchError):
        instance_generation(ctx, 0, 5, 1)  # 变换还停在 v1，BLAS 已 v2
    ok = instance_generation(ctx, 0, 5, 2)
    assert ok["mesh_version"] == 2


# --- 修复 5：同帧重建合并 -----------------------------------------------


def test_same_frame_duplicate_builds_merged():
    ctx = RTContext()
    merged = merge_frame_builds(
        ctx, [{"mesh_id": 0}, {"mesh_id": 1}, {"mesh_id": 0}], frame=1
    )
    assert merged["unique_count"] == 2 and merged["merged_count"] == 1

    b1 = build_blas(ctx, 0, frame=1)
    b2 = build_blas(ctx, 0, frame=1)  # 同帧重复 => 合并为同一次构建
    assert b1 is b2
    b3 = build_blas(ctx, 0, frame=2)  # 换帧 => 新构建
    assert b3 is not b1 and b3.version == b1.version + 1


# --- 修复 6：查询前校验实例有效 -----------------------------------------


def test_tlas_and_trace_skip_invalid_instances():
    ctx = RTContext()
    build_blas(ctx, 0, frame=1)
    instance_generation(ctx, 0, 0, 1)
    old = instance_generation(ctx, 1, 0, 1)  # 第二个实例持有旧代际
    instance_generation(ctx, 1, 0, 1, destroy=True)

    tlas = build_tlas(ctx, frame=1)
    assert [e.slot for e in tlas.entries] == [0]
    assert tlas.rejected and tlas.rejected[0]["slot"] == 1

    results = trace(ctx, [{
        "slot": 1, "generation": old["generation"],
        "origin": (0, 0, 5), "direction": (0, 0, -1),
    }])
    assert results[0]["rejected"] is True and results[0]["hit"] is False


def test_primary_and_shadow_share_same_binding_and_hit():
    ctx = RTContext()
    build_blas(ctx, 0, frame=1)
    instance_generation(ctx, 0, 0, 1)
    build_tlas(ctx, frame=1)
    ray = [{"slot": 0, "generation": 1,
            "origin": (0, 0, 5), "direction": (0, 0, -1)}]
    primary = trace(ctx, ray, "primary")
    shadow = trace(ctx, ray, "shadow")
    assert primary[0]["hit"] and shadow[0]["hit"]
    assert primary[0]["point"] == shadow[0]["point"]


# --- 兜底与全局审计 -----------------------------------------------------


def test_fallback_uses_previous_good_or_culls():
    ctx = RTContext()
    build_blas(ctx, 0, frame=1)
    instance_generation(ctx, 0, 0, 1)
    build_tlas(ctx, frame=1)
    assert fallback(ctx, 0)["strategy"] == "current"
    # 没有就绪 BLAS 的槽位直接剔除，绝不返回脏数据。
    assert fallback(ctx, 9)["strategy"] == "cull"
    assert validate_rt(ctx)["ok"] is True


def test_determinism_two_runs_identical():
    out = determinism(None, 42, 4)
    assert out["equal"] is True
    assert out["result"]["validation"]["ok"] is True
    # 场景内每帧 4 个请求含 1 个重复 => 合并计数 > 0。
    assert out["result"]["metrics"].get("builds_merged", 0) > 0
