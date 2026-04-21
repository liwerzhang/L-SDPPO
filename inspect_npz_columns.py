# # inspect_npz_columns.py
# import numpy as np, sys, os
#
# p = sys.argv[1] if len(sys.argv) > 1 else "/home/liwenzhang/Downloads/snndppo/data/robomimic/square/train.npz"
# d = np.load(p, allow_pickle=True)
# X, lens = d["states"], d["traj_lengths"]
# print("states shape:", X.shape, "traj_lengths shape:", lens.shape)
#
# # 统计每一列的基本属性
# stats = []
# for i in range(X.shape[1]):
#     v = X[:, i]
#     frac_binary = np.mean((np.isclose(v, 0.0)) | (np.isclose(v, 1.0)))
#     stats.append((i, v.mean(), v.std(), v.min(), v.max(), frac_binary))
#
# print("\nTop binary-like columns (可能是接触contact):")
# for i, m, s, mn, mx, fb in sorted(stats, key=lambda t: -t[5])[:8]:
#     print(f"  col {i:2d}: frac_0or1={fb:.3f}, mean={m:.3f}, std={s:.3f}, min={mn:.3f}, max={mx:.3f}")
#
# print("\nTop by std (变化幅度大，可能是 xyz/高度/距离):")
# for i, m, s, mn, mx, fb in sorted(stats, key=lambda t: -t[2])[:8]:
#     print(f"  col {i:2d}: std={s:.3f}, mean={m:.3f}, min={mn:.3f}, max={mx:.3f}, frac_0or1={fb:.3f}")



# import h5py
# # f = h5py.File("/home/liwenzhang/Downloads/robosuite/robosuite/models/assets/demonstrations/1762505159_5396726/demo_ld.hdf5", "r")
# # print("top groups:", list(f.keys()))           # 应该有 ['data']
# # print("data children:", list(f["data"].keys()))  # 看看是 ['demo_1', 'demo_2', ...] 还是空
# # g = list(f["data"].keys())[0]                  # 例如 'demo_1'
# # print("demo group:", g, "children:", list(f["data"][g].keys()))  # 通常只有 ['states','actions']，不会有 'obs'
# # f.close()
# f = h5py.File("/home/liwenzhang/Downloads/robosuite/robosuite/models/assets/demonstrations/1762505159_5396726/demo_ld.hdf5", "r")
#
# # with h5py.File(args.load_path, "r") as f:
# ep = sorted(list(f["data"].keys()), key=lambda s:int(s.split('_')[-1]))[0]
# print("first ep:", ep, "obs keys:", list(f["data"][ep]["obs"].keys()))


# import numpy as np
# p = "/home/liwenzhang/Downloads/snndppo/data/robomimic/door/normalization.npz"
# n = np.load(p)
# print(n.files)  # 应该包含: ['obs_min','obs_max','action_min','action_max']
# print("obs_min shape:", n['obs_min'].shape)
# print("obs_max shape:", n['obs_max'].shape)
# print("action_min shape:", n['action_min'].shape)
# print("action_max shape:", n['action_max'].shape)
#
# # 简要查看范围是否合理（min<=max，且不是NaN/Inf）
# for k in ['obs_min','obs_max','action_min','action_max']:
#     arr = n[k]
#     print(k, "min:", arr.min(), "max:", arr.max())


import numpy as np
d = np.load("/home/liwenzhang/Downloads/snndppo/data/robomimic/door/train.npz")
print(d.files)  # ['states','actions','rewards','terminals','traj_lengths']

S, A = d['states'], d['actions']
print("states range:", float(S.min()), float(S.max()))
print("actions range:", float(A.min()), float(A.max()))
