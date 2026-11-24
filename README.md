# pairwise-blas-skin-rebuild

## 领域

游戏、图形与多媒体

## 主题

蒙皮光追加速结构重建

## 说明

跨队列栅栏、实例代际与延迟释放。

初始状态下 `graphics/blas_skin_rebuild.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
