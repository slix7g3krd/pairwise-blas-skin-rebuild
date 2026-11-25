"""蒙皮光追加速结构（BLAS/TLAS）重建的 CPU 侧模拟实现。

本模块用纯 Python 状态机模拟真实图形 API（Vulkan/D3D12 风格）中的光追
加速结构管理，针对六类缺陷给出修复：

1. 跨队列栅栏：BLAS 在异步 compute 队列重建并 signal 时间线栅栏，
   graphics 队列在 build_tlas/trace 前必须 wait，杜绝读到上一帧 BLAS。
   落点：queue_fence / build_blas / build_tlas / trace / validate_rt。
2. 实例代际：TLAS 实例槽位按 slot 复用，但每次复用/销毁都递增
   generation，旧句柄校验失败，光线不会打到已失效的位置。
   落点：instance_generation / validate_instance。
3. 延迟释放：BLAS 重建后旧结构进入 pending_free，记录退役帧与栅栏值，
   只有 GPU（graphics 时间线）确实跨过该栅栏且过了退役帧后才真正回收，
   杜绝 use-after-free。落点：defer_release / release。
4. 变换与网格版本绑定：TransformBinding 不可变，内置 mesh_version，
   绑定时校验版本一致，主光线与阴影光线共用同一份绑定。
   落点：xform_bind_version / instance_generation / trace。
5. 同帧重建合并：同一帧内对同一 mesh 的多次重建请求去重，只执行一次，
   帧时间不再随角色数量线性尖刺。落点：merge_frame_builds / build_blas。
6. 查询前校验：TLAS 构建与每条光线查询前都校验实例代际/活性/BLAS
   就绪状态/版本匹配，失效实例直接剔除并计数，绝不进入遍历。
   落点：validate_instance / build_tlas / trace / fallback。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# 异常类型
# --------------------------------------------------------------------------


class RTError(Exception):
    """光追加速结构管理相关错误的基类。"""


class FenceNotWaitedError(RTError):
    """graphics 队列未等待 compute 栅栏就引用了重建结果。"""


class StaleInstanceError(RTError):
    """实例代际过期（槽位已被新角色复用或实例已销毁）。"""


class UseAfterFreeError(RTError):
    """光线查询访问了已经回收或尚未就绪的 BLAS 显存。"""


class VersionMismatchError(RTError):
    """变换绑定的 mesh_version 与当前网格版本不一致（LOD/相机切换不同步）。"""


# --------------------------------------------------------------------------
# 基础数据结构
# --------------------------------------------------------------------------


@dataclass
class Fence:
    """跨队列时间线栅栏（timeline semaphore 的一次 signal/wait 点）。"""

    name: str
    value: int
    signaled: bool = False
    waited_by_graphics: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "signaled": self.signaled,
            "waited_by_graphics": self.waited_by_graphics,
        }


@dataclass(frozen=True)
class TransformBinding:
    """不可变的变换绑定：变换矩阵与 mesh_version 原子配对。

    主光线与阴影光线必须读取同一个绑定对象，保证两套光线不会使用
    不同步的变换。
    """

    mesh_version: int
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    translate: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def matrix4x4(self) -> Tuple[float, ...]:
        sx, sy, sz = self.scale
        tx, ty, tz = self.translate
        return (
            sx, 0.0, 0.0, tx,
            0.0, sy, 0.0, ty,
            0.0, 0.0, sz, tz,
            0.0, 0.0, 0.0, 1.0,
        )

    def to_local_ray(
        self,
        origin: Sequence[float],
        direction: Sequence[float],
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """把世界空间光线变换到 BLAS 局部空间（仿射逆变换）。"""
        sx, sy, sz = self.scale
        tx, ty, tz = self.translate
        local_o = (
            (origin[0] - tx) / sx,
            (origin[1] - ty) / sy,
            (origin[2] - tz) / sz,
        )
        local_d = (direction[0] / sx, direction[1] / sy, direction[2] / sz)
        return local_o, local_d


@dataclass
class Blas:
    """底层加速结构（一个蒙皮网格某一版本的 BLAS）。"""

    mesh_id: int
    version: int
    triangles: Tuple[Tuple[Tuple[float, float, float], ...], ...]
    state: str = "building"  # building -> ready -> pending_free -> freed
    ready_fence: int = 0
    born_frame: int = 0
    retired_frame: Optional[int] = None

    @property
    def ready(self) -> bool:
        return self.state == "ready"


@dataclass
class Instance:
    """TLAS 实例：槽位 + 代际 + BLAS 引用 + 版本化变换绑定。"""

    slot: int
    generation: int
    mesh_id: int
    binding: Optional[TransformBinding]
    active: bool = True

    def handle(self) -> Tuple[int, int]:
        """外界持有的实例句柄 = (槽位, 代际)。"""
        return (self.slot, self.generation)


@dataclass
class TlasEntry:
    slot: int
    generation: int
    mesh_id: int
    blas_version: int
    binding: TransformBinding


@dataclass
class Tlas:
    frame: int
    fence_value: int
    entries: List[TlasEntry] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------
# 模拟上下文：两套队列 + 资源注册表 + 记账
# --------------------------------------------------------------------------


class RTContext:
    """持有 compute/graphics 两条队列时间线与全部光追资源状态。"""

    def __init__(self, seed: int = 0, frame: int = 0) -> None:
        self.seed = seed
        self.frame = frame
        self.rng = random.Random(seed)

        # 队列时间线：compute 只能 signal，graphics 只能 wait 后消费。
        self.compute_timeline = 0
        self.graphics_timeline = 0
        self.fences: Dict[int, Fence] = {}

        # BLAS 注册表：mesh_id -> 当前 BLAS；history 保留所有代（含待释放）。
        self.blas_current: Dict[int, Blas] = {}
        self.blas_history: List[Blas] = []

        # TLAS 实例槽位及其代际历史。
        self.instances: Dict[int, Instance] = {}
        self.slot_generations: Dict[int, List[int]] = {}

        # 延迟释放队列：(退役帧, 栅栏值, 资源)。
        self.pending_release: List[Tuple[int, int, Blas]] = []

        # 同帧重建去重：frame -> set(mesh_id)，只记录本帧已实际执行的构建。
        self.frame_builds: Dict[int, set] = {}
        # merge_frame_builds 产出的本帧去重计划（不代表已执行）。
        self.frame_plan: Dict[int, List[Any]] = {}

        # 上一帧确认有效的实例引用，供 fallback 使用。
        self.last_good: Dict[int, TlasEntry] = {}

        self.last_tlas: Optional[Tlas] = None
        self.audit_log: List[Dict[str, Any]] = []
        self.metrics: Dict[str, int] = {}

    # --- 记账辅助 ---

    def count(self, name: str, n: int = 1) -> None:
        self.metrics[name] = self.metrics.get(name, 0) + n

    def event(self, kind: str, **payload: Any) -> None:
        record = {"frame": self.frame, "event": kind}
        record.update(payload)
        self.audit_log.append(record)

    def enter_frame(self, frame: int) -> None:
        """推进到指定帧：同帧去重集合只保留当前帧，历史帧记录清掉。

        没有这个轮转，frame N 的“已重建”标记会错误命中 frame N+1 的
        正常重建请求，导致跨帧合并（真实 bug：重建被吞、BLAS 不更新）。
        """
        if frame == self.frame:
            return
        if frame < self.frame:
            raise RTError(f"帧号单调递增: {frame} < {self.frame}")
        self.frame = frame
        for store in (self.frame_builds, self.frame_plan):
            for old in [f for f in store if f != frame]:
                del store[old]

    # --- 队列时间线操作 ---

    def compute_signal(self, value: Optional[int] = None) -> Fence:
        if value is None:
            value = self.compute_timeline + 1
        if value <= self.compute_timeline:
            raise RTError(f"栅栏值必须单调递增: {value} <= {self.compute_timeline}")
        fence = Fence(name=f"compute->graphics@{value}", value=value, signaled=True)
        self.fences[value] = fence
        self.compute_timeline = value
        self.count("fences_signaled")
        self.event("fence_signal", queue="compute", value=value)
        return fence

    def graphics_wait(self, value: int) -> Fence:
        if value > self.compute_timeline:
            raise FenceNotWaitedError(
                f"graphics 试图 wait {value}，但 compute 仅 signal 到 "
                f"{self.compute_timeline}：跨队列栅栏缺失"
            )
        fence = self.fences[value]
        fence.waited_by_graphics = True
        if value > self.graphics_timeline:
            self.graphics_timeline = value
            self.count("fence_waits")
        self.event("fence_wait", queue="graphics", value=value)
        return fence


# --------------------------------------------------------------------------
# 通用辅助
# --------------------------------------------------------------------------


def _coerce_ctx(arg: Any) -> RTContext:
    """允许公共函数以 RTContext、帧号（int）或 None 作为第一个参数。"""
    if isinstance(arg, RTContext):
        return arg
    ctx = RTContext()
    if isinstance(arg, int):
        ctx.frame = arg
    return ctx


def _arg(
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    idx: int,
    name: str,
    default: Any = None,
) -> Any:
    if name in kwargs:
        return kwargs[name]
    if idx < len(args):
        return args[idx]
    return default


_VEC = Tuple[float, float, float]


def _sub(a: _VEC, b: _VEC) -> _VEC:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: _VEC, b: _VEC) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: _VEC, b: _VEC) -> _VEC:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _ray_triangle(
    origin: _VEC,
    direction: _VEC,
    v0: _VEC,
    v1: _VEC,
    v2: _VEC,
) -> Optional[float]:
    """Möller–Trumbore 求交，返回射线参数 t；不相交返回 None。"""
    eps = 1e-9
    e1 = _sub(v1, v0)
    e2 = _sub(v2, v0)
    p = _cross(direction, e2)
    det = _dot(e1, p)
    if abs(det) < eps:
        return None
    tvec = _sub(origin, v0)
    u = _dot(tvec, p) / det
    if not 0.0 <= u <= 1.0:
        return None
    q = _cross(tvec, e1)
    v = _dot(direction, q) / det
    if v < 0.0 or u + v > 1.0:
        return None
    t = _dot(e2, q) / det
    return t if t > eps else None


def _skin_geometry(
    mesh_id: int,
    version: int,
) -> Tuple[Tuple[_VEC, ...], Tuple[Tuple[int, int, int], ...]]:
    """确定性生成某 mesh 某版本的蒙皮几何体（版本不同 => 顶点形变不同）。"""
    rng = random.Random((mesh_id * 1_000_003) ^ (version * 7919) ^ 0x5EED)
    cube = [
        (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5),
    ]
    # 蒙皮形变：整体随版本平移 + 每顶点确定性抖动，模拟每帧烘焙的蒙皮结果。
    verts = tuple(
        (
            x + version * 0.25 + rng.uniform(-0.01, 0.01),
            y + rng.uniform(-0.01, 0.01),
            z + rng.uniform(-0.01, 0.01),
        )
        for (x, y, z) in cube
    )
    faces = (
        (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
        (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0),
    )
    return verts, faces


def _build_triangles(mesh_id: int, version: int):
    verts, faces = _skin_geometry(mesh_id, version)
    return tuple(tuple(verts[i] for i in face) for face in faces)


# ==========================================================================
# 修复 1：跨队列插栅栏
# ==========================================================================


def queue_fence(ctx: Any = None, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """在 compute 与 graphics 队列之间插入时间线栅栏。

    op="signal"（默认）：compute 队列重建完成后 signal；
    op="wait"：graphics 队列在引用 BLAS/TLAS 前 wait，未 signal 直接抛错。
    """
    ctx = _coerce_ctx(ctx)
    value = _arg(args, kwargs, 0, "value", None)
    op = _arg(args, kwargs, 1, "op", "signal")

    if op == "wait":
        if value is None:
            value = ctx.compute_timeline
        fence = ctx.graphics_wait(int(value))
    else:
        fence = ctx.compute_signal(None if value is None else int(value))
    return fence.as_dict()


# ==========================================================================
# 修复 2：实例按代际标记
# ==========================================================================


def instance_generation(ctx: Any = None, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """分配/复用/销毁一个 TLAS 实例槽位，每次都递增代际。

    槽位可以被新角色复用，但 generation 单调递增；外界持有的旧
    (slot, generation) 句柄此后永远校验失败。
    """
    ctx = _coerce_ctx(ctx)
    slot = int(_arg(args, kwargs, 0, "slot", 0))
    mesh_id = _arg(args, kwargs, 1, "mesh_id", slot)
    mesh_version = _arg(args, kwargs, 2, "mesh_version", None)
    destroy = bool(kwargs.get("destroy", False))
    translate = kwargs.get("translate", (0.0, 0.0, 0.0))
    scale = kwargs.get("scale", (1.0, 1.0, 1.0))

    previous = ctx.instances.get(slot)
    generation = previous.generation + 1 if previous is not None else 1
    ctx.slot_generations.setdefault(slot, []).append(generation)

    binding: Optional[TransformBinding] = None
    if not destroy:
        version = int(mesh_version) if mesh_version is not None else 0
        binding = xform_bind_version(ctx, version, None, translate=translate, scale=scale)
        # 版本绑定校验：BLAS 已存在时，变换版本必须与当前网格版本一致。
        blas = ctx.blas_current.get(int(mesh_id))
        if blas is not None and blas.version != version:
            raise VersionMismatchError(
                f"slot {slot} 绑定 mesh_version={version}，但 mesh {mesh_id} "
                f"当前 BLAS 版本为 {blas.version}（相机/LOD 切换不同步）"
            )

    instance = Instance(
        slot=slot,
        generation=generation,
        mesh_id=int(mesh_id),
        binding=binding,
        active=not destroy,
    )
    ctx.instances[slot] = instance
    ctx.count("instances_created")
    if previous is not None:
        ctx.count("generations_bumped")
    ctx.event(
        "instance_generation",
        slot=slot,
        generation=generation,
        mesh_id=int(mesh_id),
        active=not destroy,
        destroyed=destroy,
    )
    return {
        "slot": slot,
        "generation": generation,
        "mesh_id": int(mesh_id),
        "active": instance.active,
        "binding": None if binding is None else binding.matrix4x4(),
        "mesh_version": None if binding is None else binding.mesh_version,
    }


# ==========================================================================
# 修复 3：延迟释放旧结构
# ==========================================================================


def defer_release(ctx: Any = None, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """把旧 BLAS 放入延迟释放队列，而不是立即回收显存。

    记录退役帧与当前 compute 栅栏值；release() 只有在 graphics 时间线
    跨过该栅栏且进入更晚帧之后才真正 free。
    """
    ctx = _coerce_ctx(ctx)
    resource = _arg(args, kwargs, 0, "resource", None)
    fence_value = _arg(args, kwargs, 1, "fence_value", ctx.compute_timeline)

    if resource is None:
        return {
            "pending": [
                {
                    "mesh_id": b.mesh_id,
                    "version": b.version,
                    "state": b.state,
                    "retired_frame": f,
                    "fence_value": v,
                }
                for (f, v, b) in ctx.pending_release
            ],
            "count": len(ctx.pending_release),
        }

    blas = resource if isinstance(resource, Blas) else resource
    blas.state = "pending_free"
    blas.retired_frame = ctx.frame
    ctx.pending_release.append((ctx.frame, int(fence_value), blas))
    ctx.count("deferred_frees")
    ctx.event(
        "defer_release",
        mesh_id=blas.mesh_id,
        version=blas.version,
        fence_value=int(fence_value),
    )
    return {
        "mesh_id": blas.mesh_id,
        "version": blas.version,
        "state": "pending_free",
        "retired_frame": ctx.frame,
        "fence_value": int(fence_value),
    }


def release(ctx: Any = None, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """回收满足条件的延迟释放资源：栅栏已被 graphics 跨过且跨帧。"""
    ctx = _coerce_ctx(ctx)
    frame = _arg(args, kwargs, 0, "frame", None)
    if frame is not None:
        ctx.enter_frame(int(frame))

    still_pending: List[Tuple[int, int, Blas]] = []
    freed: List[Dict[str, Any]] = []
    for retired_frame, fence_value, blas in ctx.pending_release:
        fence_done = ctx.graphics_timeline >= fence_value
        frame_passed = ctx.frame > retired_frame
        if fence_done and frame_passed:
            blas.state = "freed"
            freed.append({"mesh_id": blas.mesh_id, "version": blas.version})
            ctx.count("frees_completed")
            ctx.event(
                "release",
                mesh_id=blas.mesh_id,
                version=blas.version,
                fence_value=fence_value,
            )
        else:
            still_pending.append((retired_frame, fence_value, blas))
    ctx.pending_release = still_pending
    return {"freed": freed, "freed_count": len(freed),
            "still_pending": len(still_pending)}


# ==========================================================================
# 修复 4：变换与网格版本绑定
# ==========================================================================


def xform_bind_version(ctx: Any = None, *args: Any, **kwargs: Any) -> TransformBinding:
    """创建不可变的 (变换, mesh_version) 绑定。

    expected_version 给定时做一致性断言：相机/LOD 切换中若变换对应的
    网格版本与实际版本不符，直接抛 VersionMismatchError，而不是带着
    错版本继续渲染。
    """
    ctx = _coerce_ctx(ctx)
    mesh_version = int(_arg(args, kwargs, 0, "mesh_version", 0))
    matrix = _arg(args, kwargs, 1, "matrix", None)
    translate = kwargs.get("translate", None)
    scale = kwargs.get("scale", None)
    expected = kwargs.get("expected_version", None)

    if expected is not None and int(expected) != mesh_version:
        raise VersionMismatchError(
            f"变换绑定版本 {mesh_version} 与期望网格版本 {expected} 不一致"
        )

    if matrix is not None:
        # 仅接受仿射 4x4：取对角线缩放与平移列。
        if len(matrix) != 16:
            raise RTError("matrix 必须是 16 元素行主序 4x4 矩阵")
        scale = (matrix[0], matrix[5], matrix[10]) if scale is None else scale
        translate = (matrix[3], matrix[7], matrix[11]) if translate is None else translate
    if translate is None:
        translate = (0.0, 0.0, 0.0)
    if scale is None:
        scale = (1.0, 1.0, 1.0)

    binding = TransformBinding(
        mesh_version=mesh_version,
        scale=tuple(float(x) for x in scale),
        translate=tuple(float(x) for x in translate),
    )
    ctx.event("xform_bind_version", mesh_version=mesh_version)
    return binding


# ==========================================================================
# 修复 5：同帧重建合并
# ==========================================================================


def merge_frame_builds(ctx: Any = None, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """把同一帧的重建请求按 mesh 去重，保留首次请求顺序。

    本函数只产出去重后的构建计划（frame_plan），不标记任何 BLAS 为
    “已构建”；真正的同帧幂等由 build_blas 依据 frame_builds 执行集合
    保证，避免“计划被当成执行”吞掉后续帧的正常重建。
    """
    ctx = _coerce_ctx(ctx)
    requests = _arg(args, kwargs, 0, "requests", [])
    frame = _arg(args, kwargs, 1, "frame", None)
    if frame is not None:
        ctx.enter_frame(int(frame))

    seen: set = set()
    merged: List[Any] = []
    duplicate_count = 0
    for req in requests:
        mesh_id = req["mesh_id"] if isinstance(req, dict) else int(req)
        if mesh_id in seen:
            duplicate_count += 1
            continue
        seen.add(mesh_id)
        merged.append(req)
    ctx.frame_plan[ctx.frame] = merged
    ctx.count("build_requests", len(requests))
    ctx.count("builds_merged", duplicate_count)
    ctx.event(
        "merge_frame_builds",
        frame=ctx.frame,
        requested=len(requests),
        unique=len(merged),
        merged=duplicate_count,
    )
    return {
        "frame": ctx.frame,
        "requests": merged,
        "unique_count": len(merged),
        "merged_count": duplicate_count,
    }


# ==========================================================================
# 修复 6：查询前校验实例有效
# ==========================================================================


def validate_instance(ctx: Any = None, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """校验实例句柄：槽位存在、代际匹配、未销毁、BLAS 就绪、版本一致。"""
    ctx = _coerce_ctx(ctx)
    slot = int(_arg(args, kwargs, 0, "slot", 0))
    generation = _arg(args, kwargs, 1, "generation", None)

    reasons: List[str] = []
    instance = ctx.instances.get(slot)
    if instance is None:
        reasons.append("slot_not_allocated")
        return {"valid": False, "slot": slot, "reasons": reasons}

    if generation is not None and instance.generation != int(generation):
        reasons.append(
            f"stale_generation(handle={generation}, current={instance.generation})"
        )
    if not instance.active:
        reasons.append("instance_destroyed")
    if instance.binding is None:
        reasons.append("missing_binding")

    blas = ctx.blas_current.get(instance.mesh_id)
    if blas is None:
        reasons.append("blas_missing")
    elif blas.state != "ready":
        reasons.append(f"blas_not_ready(state={blas.state})")
    elif instance.binding is not None and blas.version != instance.binding.mesh_version:
        reasons.append(
            f"version_mismatch(blas={blas.version}, binding={instance.binding.mesh_version})"
        )

    valid = not reasons
    if not valid:
        ctx.count("invalid_validations")
    return {
        "valid": valid,
        "slot": slot,
        "generation": instance.generation,
        "mesh_id": instance.mesh_id,
        "reasons": reasons,
    }


# ==========================================================================
# BLAS / TLAS 构建与光线查询
# ==========================================================================


def build_blas(ctx: Any = None, *args: Any, **kwargs: Any) -> Blas:
    """在 compute 队列重建一个蒙皮 mesh 的 BLAS。

    - 同帧同 mesh 的重复调用直接合并，返回本帧已建 BLAS；
    - 旧 BLAS 不立即释放，进入延迟释放队列；
    - 构建完成后 compute signal 栅栏，graphics 必须 wait 才能引用。
    """
    ctx = _coerce_ctx(ctx)
    mesh_id = int(_arg(args, kwargs, 0, "mesh_id", 0))
    frame = _arg(args, kwargs, 1, "frame", None)
    mesh_version = _arg(args, kwargs, 2, "mesh_version", None)
    if frame is not None:
        ctx.enter_frame(int(frame))

    # 修复 5：同帧重建合并。
    seen = ctx.frame_builds.setdefault(ctx.frame, set())
    if mesh_id in seen and mesh_id in ctx.blas_current:
        ctx.count("builds_merged")
        ctx.event("build_blas_merged", mesh_id=mesh_id, frame=ctx.frame)
        return ctx.blas_current[mesh_id]
    seen.add(mesh_id)

    old = ctx.blas_current.get(mesh_id)
    version = old.version + 1 if old is not None else 1
    if mesh_version is not None:
        version = int(mesh_version)

    fence_value = ctx.compute_timeline + 1
    blas = Blas(
        mesh_id=mesh_id,
        version=version,
        triangles=_build_triangles(mesh_id, version),
        state="building",
        ready_fence=fence_value,
        born_frame=ctx.frame,
    )
    ctx.blas_history.append(blas)
    ctx.blas_current[mesh_id] = blas
    ctx.count("blas_builds")

    # compute 队列执行构建并 signal（模拟 GPU 执行完毕）。
    blas.state = "ready"
    ctx.compute_signal(fence_value)
    ctx.event(
        "build_blas",
        mesh_id=mesh_id,
        version=version,
        fence_value=fence_value,
        frame=ctx.frame,
    )

    # 修复 3：旧结构延迟释放，绝不立即回收。
    if old is not None and old.state == "ready":
        defer_release(ctx, old, fence_value)

    return blas


def build_tlas(ctx: Any = None, *args: Any, **kwargs: Any) -> Tlas:
    """graphics 队列构建 TLAS：先跨队列 wait，再逐实例校验后收录。"""
    ctx = _coerce_ctx(ctx)
    frame = _arg(args, kwargs, 0, "frame", None)
    if frame is not None:
        ctx.enter_frame(int(frame))

    # 修复 1：引用任何 compute 重建结果前先 wait 栅栏。
    fence_value = ctx.compute_timeline
    if fence_value > 0:
        ctx.graphics_wait(fence_value)

    tlas = Tlas(frame=ctx.frame, fence_value=fence_value)
    for slot, instance in sorted(ctx.instances.items()):
        check = validate_instance(ctx, slot, instance.generation)
        if not check["valid"]:
            tlas.rejected.append({"slot": slot, "reasons": check["reasons"]})
            ctx.count("tlas_instances_rejected")
            ctx.event("tlas_reject", slot=slot, reasons=check["reasons"])
            continue
        blas = ctx.blas_current[instance.mesh_id]
        entry = TlasEntry(
            slot=slot,
            generation=instance.generation,
            mesh_id=instance.mesh_id,
            blas_version=blas.version,
            binding=instance.binding,
        )
        tlas.entries.append(entry)
        ctx.last_good[slot] = entry
    ctx.last_tlas = tlas
    ctx.count("tlas_builds")
    ctx.event(
        "build_tlas",
        frame=ctx.frame,
        entries=len(tlas.entries),
        rejected=len(tlas.rejected),
    )
    return tlas


def _default_rays(ctx: RTContext, entries: List[TlasEntry]) -> List[Dict[str, Any]]:
    if not entries:
        return []
    rays: List[Dict[str, Any]] = []
    for entry in entries:
        # 从 +Z 远处朝原点发射，确定性由 ctx.rng 控制。
        jitter = ctx.rng.uniform(-0.05, 0.05)
        rays.append({
            "slot": entry.slot,
            "generation": entry.generation,
            "origin": (jitter, jitter, 5.0),
            "direction": (0.0, 0.0, -1.0),
        })
    return rays


def _trace_one(entry: TlasEntry, blas: Blas, ray: Dict[str, Any]) -> Optional[float]:
    origin = ray["origin"]
    direction = ray["direction"]
    local_o, local_d = entry.binding.to_local_ray(origin, direction)
    best: Optional[float] = None
    for tri in blas.triangles:
        t = _ray_triangle(local_o, local_d, tri[0], tri[1], tri[2])
        if t is not None and (best is None or t < best):
            best = t
    return best


def trace(ctx: Any = None, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
    """在 graphics 队列执行光线查询。

    每条光线查询前重新校验实例（修复 6），并断言 BLAS 处于 ready 态
    （修复 3 的访问侧护栏，触发即 UseAfterFreeError）；主光线与阴影
    光线共用同一份版本化变换绑定（修复 4）。
    """
    ctx = _coerce_ctx(ctx)
    rays = _arg(args, kwargs, 0, "rays", None)
    kind = _arg(args, kwargs, 1, "kind", "primary")

    if ctx.compute_timeline > ctx.graphics_timeline:
        ctx.graphics_wait(ctx.compute_timeline)

    tlas = ctx.last_tlas
    if tlas is None:
        tlas = build_tlas(ctx)
    if rays is None:
        rays = _default_rays(ctx, tlas.entries)

    results: List[Dict[str, Any]] = []
    for ray in rays:
        slot = ray["slot"]
        generation = ray.get("generation")

        # 显存安全优先：只要实例指向的 BLAS 不处于 ready，硬拦截，
        # 绝不允许后续任何路径访问 building/pending_free/freed 显存。
        instance = ctx.instances.get(slot)
        if instance is not None:
            blas = ctx.blas_current.get(instance.mesh_id)
            if blas is not None and blas.state != "ready":
                raise UseAfterFreeError(
                    f"slot {slot} 的 BLAS(mesh={blas.mesh_id}, v{blas.version}) "
                    f"状态为 {blas.state}，光线查询被护栏拦截"
                )

        check = validate_instance(ctx, slot, generation)
        if not check["valid"]:
            ctx.count("queries_rejected_invalid_instance")
            ctx.event("trace_reject", slot=slot, reasons=check["reasons"])
            results.append({
                "slot": slot, "kind": kind, "hit": False,
                "rejected": True, "reasons": check["reasons"],
            })
            continue

        instance = ctx.instances[slot]
        blas = ctx.blas_current[instance.mesh_id]
        if blas.state != "ready":
            # 双保险：上面的快速护栏漏过时在此再次拦截。
            raise UseAfterFreeError(
                f"slot {slot} 的 BLAS(mesh={blas.mesh_id}, v{blas.version}) "
                f"状态为 {blas.state}，光线查询被护栏拦截"
            )

        entry = next((e for e in tlas.entries if e.slot == slot), None)
        if entry is None or entry.generation != instance.generation:
            ctx.count("queries_rejected_invalid_instance")
            results.append({
                "slot": slot, "kind": kind, "hit": False,
                "rejected": True, "reasons": ["not_in_tlas"],
            })
            continue

        t = _trace_one(entry, blas, ray)
        ctx.count("rays_traced")
        hit = t is not None
        if hit:
            ctx.count("ray_hits")
            o, d = ray["origin"], ray["direction"]
            world_point = tuple(o[i] + t * d[i] for i in range(3))
            results.append({
                "slot": slot,
                "kind": kind,
                "hit": True,
                "t": t,
                "point": world_point,
                "mesh_id": blas.mesh_id,
                "blas_version": blas.version,
                "mesh_version": entry.binding.mesh_version,
            })
        else:
            results.append({"slot": slot, "kind": kind, "hit": False})
    ctx.event("trace", ray_kind=kind, rays=len(rays))
    return results


# ==========================================================================
# 全局校验 / 可观测性 / 兜底 / 确定性
# ==========================================================================


def validate_rt(ctx: Any = None, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """对当前加速结构状态做完整性审计，返回所有违规项。"""
    ctx = _coerce_ctx(ctx)
    violations: List[str] = []

    # 1. 活动实例引用的 BLAS 必须存在、ready、版本匹配。
    for slot, inst in ctx.instances.items():
        if not inst.active:
            continue
        blas = ctx.blas_current.get(inst.mesh_id)
        if blas is None:
            violations.append(f"slot {slot}: blas missing for mesh {inst.mesh_id}")
            continue
        if blas.state == "freed":
            violations.append(f"slot {slot}: references freed BLAS (use-after-free)")
        elif blas.state != "ready":
            violations.append(f"slot {slot}: BLAS state={blas.state}")
        if inst.binding is not None and blas.version != inst.binding.mesh_version:
            violations.append(
                f"slot {slot}: transform mesh_version={inst.binding.mesh_version} "
                f"!= blas version={blas.version}"
            )

    # 2. TLAS 收录的 BLAS 都必须已被 graphics 栅栏消费，且不在待释放队列。
    pending_ids = {(b.mesh_id, b.version) for (_, _, b) in ctx.pending_release}
    if ctx.last_tlas is not None:
        tlas = ctx.last_tlas
        if tlas.fence_value > ctx.graphics_timeline:
            violations.append(
                f"TLAS fence {tlas.fence_value} 未被 graphics wait "
                f"(timeline={ctx.graphics_timeline})"
            )
        for entry in tlas.entries:
            if (entry.mesh_id, entry.blas_version) in pending_ids:
                violations.append(
                    f"TLAS slot {entry.slot} references pending-free BLAS "
                    f"mesh {entry.mesh_id} v{entry.blas_version}"
                )
            if entry.binding.mesh_version != entry.blas_version:
                violations.append(
                    f"TLAS slot {entry.slot}: binding/BLAS version mismatch"
                )

    # 3. 每槽位代际单调递增。
    for slot, generations in ctx.slot_generations.items():
        if generations != sorted(generations) or len(set(generations)) != len(generations):
            violations.append(f"slot {slot}: generation not monotonic {generations}")

    # 4. 延迟释放队列中不得出现 freed 资源。
    for _, _, blas in ctx.pending_release:
        if blas.state == "freed":
            violations.append(f"freed BLAS still in release queue: mesh {blas.mesh_id}")

    return {
        "ok": not violations,
        "violations": violations,
        "instances": len(ctx.instances),
        "blas_total": len(ctx.blas_history),
        "pending_release": len(ctx.pending_release),
    }


def metrics(ctx: Any = None, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """返回计数器快照与派生指标。"""
    ctx = _coerce_ctx(ctx)
    out = dict(ctx.metrics)
    requested = out.get("build_requests", out.get("blas_builds", 0))
    builds = out.get("blas_builds", 0)
    out["merge_rate"] = (
        out.get("builds_merged", 0) / requested if requested else 0.0
    )
    out["blas_live"] = sum(1 for b in ctx.blas_history if b.state == "ready")
    out["blas_pending_free"] = sum(
        1 for b in ctx.blas_history if b.state == "pending_free"
    )
    out["blas_freed"] = sum(1 for b in ctx.blas_history if b.state == "freed")
    out["active_instances"] = sum(1 for i in ctx.instances.values() if i.active)
    return out


def audit(ctx: Any = None, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
    """返回事件审计日志（栅栏、构建、代际、释放、拒绝、兜底等）。"""
    ctx = _coerce_ctx(ctx)
    return list(ctx.audit_log)


def fallback(ctx: Any = None, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """BLAS 未就绪时的兜底策略：用上一帧确认有效的 BLAS，否则剔除实例。

    永远不返回 freed / building 状态的 BLAS，保证不读错位或已回收数据。
    """
    ctx = _coerce_ctx(ctx)
    slot = int(_arg(args, kwargs, 0, "slot", 0))
    instance = ctx.instances.get(slot)

    if instance is None or not instance.active:
        result = {"slot": slot, "strategy": "cull", "reason": "invalid_or_destroyed"}
    else:
        blas = ctx.blas_current.get(instance.mesh_id)
        fence_ready = blas is not None and blas.state == "ready" and (
            blas.ready_fence <= ctx.graphics_timeline
        )
        if fence_ready:
            result = {
                "slot": slot,
                "strategy": "current",
                "mesh_id": blas.mesh_id,
                "blas_version": blas.version,
            }
        elif slot in ctx.last_good:
            good = ctx.last_good[slot]
            # 二次确认 last_good 指向的 BLAS 仍未被回收。
            ref = next(
                (b for b in ctx.blas_history
                 if b.mesh_id == good.mesh_id and b.version == good.blas_version),
                None,
            )
            if ref is not None and ref.state == "ready":
                result = {
                    "slot": slot,
                    "strategy": "previous_blas",
                    "mesh_id": good.mesh_id,
                    "blas_version": good.blas_version,
                }
            else:
                result = {"slot": slot, "strategy": "cull", "reason": "last_good_recycled"}
        else:
            result = {"slot": slot, "strategy": "cull", "reason": "no_ready_blas"}

    ctx.count("fallbacks")
    ctx.event("fallback", **result)
    return result


def _scenario(seed: int, frames: int) -> Dict[str, Any]:
    ctx = RTContext(seed=seed)
    for frame in range(1, frames + 1):
        # 同帧含重复请求（mesh 0 出现两次），验证合并。
        merged = merge_frame_builds(ctx, [0, 1, 0, 2], frame=frame)
        for req in merged["requests"]:
            mesh_id = req["mesh_id"] if isinstance(req, dict) else req
            build_blas(ctx, mesh_id, frame=frame, mesh_version=frame)

        def bind(slot: int, mesh: int, **extra: Any) -> None:
            # 变换始终绑定当前 BLAS 的真实版本（修复 4：版本原子配对）。
            current_version = ctx.blas_current[mesh].version
            instance_generation(
                ctx, slot, mesh, current_version,
                translate=(float(slot), 0.0, 0.0), **extra,
            )

        if frame == 1:
            for slot, mesh_id in ((0, 0), (1, 1), (2, 2)):
                bind(slot, mesh_id)
        else:
            # 每帧用最新蒙皮版本重新绑定活动实例。
            for slot, mesh_id in ((0, 0), (1, 1)):
                bind(slot, mesh_id)
            if frame == 2:
                # 销毁 slot 2 并立刻由新角色复用：代际递增，旧句柄失效。
                instance_generation(ctx, 2, 1, frame, destroy=True)
                bind(2, 1)
            else:
                bind(2, 1)

        queue_fence(ctx)
        build_tlas(ctx, frame=frame)
        primary = trace(ctx, None, "primary")
        shadow = trace(ctx, None, "shadow")
        release(ctx, frame=frame)

    return {
        "metrics": metrics(ctx),
        "tlas": [
            {
                "slot": e.slot,
                "generation": e.generation,
                "mesh_id": e.mesh_id,
                "blas_version": e.blas_version,
                "binding": e.binding.matrix4x4(),
            }
            for e in ctx.last_tlas.entries
        ],
        "rejected": ctx.last_tlas.rejected,
        "validation": validate_rt(ctx),
        "primary_sample": trace(ctx, None, "primary"),
        "shadow_sample": trace(ctx, None, "shadow"),
        "audit_tail": ctx.audit_log[-6:],
    }


def determinism(ctx: Any = None, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """同种子同输入跑两遍场景，构建/TLAS/命中/记账必须逐字节一致。"""
    seed = _arg(args, kwargs, 0, "seed", 0)
    frames = int(_arg(args, kwargs, 1, "frames", 4))
    if seed is None:
        seed = ctx.seed if isinstance(ctx, RTContext) else 0
    seed = int(seed)

    run_a = _scenario(seed, frames)
    run_b = _scenario(seed, frames)
    return {
        "equal": run_a == run_b,
        "seed": seed,
        "frames": frames,
        "result": run_a,
    }


__all__ = [
    # 六个修复点入口
    "queue_fence",
    "instance_generation",
    "defer_release",
    "xform_bind_version",
    "merge_frame_builds",
    "validate_instance",
    # 管线入口
    "build_blas",
    "build_tlas",
    "trace",
    "release",
    # 质量保障入口
    "validate_rt",
    "metrics",
    "audit",
    "fallback",
    "determinism",
    # 数据结构与异常
    "RTContext",
    "Fence",
    "Blas",
    "Instance",
    "Tlas",
    "TransformBinding",
    "RTError",
    "FenceNotWaitedError",
    "StaleInstanceError",
    "UseAfterFreeError",
    "VersionMismatchError",
]
