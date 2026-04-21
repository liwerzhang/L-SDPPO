# npz_audit_grasp_lift.py  (with SwanLab / W&B logging)
import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt

def split_episodes(X, lens):
    ep = []
    s = 0
    for L in lens:
        ep.append(X[s:s+L]); s += L
    return ep

def suggest_gripper_cols(X, topk=6):
    stats = []
    for i in range(X.shape[1]):
        v = X[:, i]
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        near_n1 = np.mean(np.isclose(v, -1.0, atol=0.1))
        near_p1 = np.mean(np.isclose(v,  1.0, atol=0.1))
        span    = np.percentile(v, 95) - np.percentile(v, 5)
        std     = float(np.std(v))
        score   = (near_n1 + near_p1) + 0.3*span + 0.1*std
        stats.append((score, i, near_n1, near_p1, span, std))
    stats.sort(reverse=True)
    return stats[:topk]

def detect_episode(gc_bool, z, n_grasp=6, h_min_norm=-0.4, m_lift=6):
    T = min(len(gc_bool), len(z))
    run = 0; ge = -1
    for t in range(T):
        run = run + 1 if gc_bool[t] else 0
        if run >= n_grasp:
            ge = t; break
    grasp_ok = ge >= 0

    lift_ok, ls, le = False, -1, -1
    if grasp_ok:
        run = 0
        for t in range(ge+1, T):
            ok = z[t] >= h_min_norm
            run = run + 1 if ok else 0
            if ok and run == 1: ls = t
            if run >= m_lift: le = t; lift_ok = True; break

    max_lift = float(np.max(z[:T])) if T>0 else 0.0
    return dict(grasp_ok=grasp_ok, grasp_end=ge, lift_ok=lift_ok,
                lift_start=ls, lift_end=le, max_lift=max_lift)

def plot_episode(ep_idx, z, gc_val, gc_bool, det, h_min_norm, save_dir):
    t = np.arange(len(z))
    fig, axes = plt.subplots(3,1, figsize=(8,6), sharex=True)
    axes[0].plot(t, z); axes[0].axhline(h_min_norm, ls="--"); axes[0].set_ylabel("z (norm)")
    axes[1].plot(t, gc_val); axes[1].set_ylabel("gripper(raw)")
    axes[2].plot(t, gc_bool.astype(float)); axes[2].set_ylabel("gripper(closed)")
    axes[2].set_xlabel("timestep")
    if det["grasp_ok"]:
        ge = det["grasp_end"]; gs = max(0, ge-5)
        for ax in axes: ax.axvspan(gs, ge, alpha=0.15)
    if det["lift_ok"]:
        for ax in axes: ax.axvspan(det["lift_start"], det["lift_end"], alpha=0.15)
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"ep_{ep_idx:04d}.png")
    fig.savefig(path)
    plt.close(fig)
    return path  # 返回图片路径，便于日志上报

def setup_logger(args):
    """
    返回 (wb, run)。wb 为 wandb 模块或 None；run 为 wandb.Run 或 None。
    --log none|wandb|swanlab
    """
    if args.log == "none":
        return None, None
    if args.log == "wandb":
        import wandb as wb
        run = wb.init(project=args.project, name=args.run_name, config=vars(args), reinit=True)
        return wb, run
    if args.log == "swanlab":
        import swanlab
        swanlab.sync_wandb()     # 让 W&B API 事件转发到 SwanLab
        import wandb as wb
        run = wb.init(project=args.project, name=args.run_name, config=vars(args), reinit=True)
        return wb, run
    raise ValueError("--log must be one of: none, wandb, swanlab")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="path to train.npz")
    ap.add_argument("--z-col", type=int, default=18, help="方螺母 z 在 states 的列索引")
    ap.add_argument("--h-min-norm", type=float, required=True, help="归一化后的抬起阈值")
    ap.add_argument("--n-grasp", type=int, default=6, help="抓稳 连续步数")
    ap.add_argument("--m-lift",  type=int, default=6, help="抬起 连续步数")
    ap.add_argument("--grip-col", type=int, default=None, help="夹爪列（不指定则自动猜测）")
    ap.add_argument("--grip-close-thr", type=float, default=0.0,
                    help="gripper_raw <= 该阈值 视为闭合；若你的数据是 1=闭合，可设 0.5 或改比较方向")
    ap.add_argument("--plot-k", type=int, default=8, help="最多可视化多少条示例")
    ap.add_argument("--out-dir", default="npz_audit_out", help="本地输出目录")
    # 日志相关
    ap.add_argument("--log", choices=["none","wandb","swanlab"], default="none")
    ap.add_argument("--project", default="nut-assembly-audit")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    wb, run = setup_logger(args)

    data = np.load(args.npz, allow_pickle=True)
    X, lens = data["states"], data["traj_lengths"]
    episodes = split_episodes(X, lens)

    # 自动猜测夹爪列
    grip_col = args.grip_col
    if grip_col is None:
        cand = suggest_gripper_cols(X, topk=6)
        print("\n[Suggest gripper candidates]:")
        for sc, i, n1, p1, span, sd in cand:
            print(f"  col {i:2d}: score={sc:.3f}, near-(-1,1)=({n1:.2f},{p1:.2f}), span={span:.2f}, std={sd:.2f}")
        grip_col = cand[0][1] if cand else 0
        print(f"=> Use grip_col = {grip_col}\n")
    else:
        print(f"=> Use user-specified grip_col = {grip_col}\n")

    ok_g, ok_gl, total = 0, 0, 0
    plot_dir = os.path.join(args.out_dir, "plots")
    for epi, E in enumerate(episodes):
        z  = E[:, args.z_col]
        gr = E[:, grip_col]
        gr_closed = gr <= args.grip_close_thr   # 如需“>=”请改这行

        det = detect_episode(gr_closed, z,
                             n_grasp=args.n_grasp,
                             h_min_norm=args.h_min_norm,
                             m_lift=args.m_lift)
        ok_g  += int(det["grasp_ok"])
        ok_gl += int(det["lift_ok"])
        total += 1

        img_path = None
        if epi < args.plot_k:
            img_path = plot_episode(epi, z, gr, gr_closed.astype(int), det,
                                    args.h_min_norm, plot_dir)

        # 日志上报（按集）
        if wb is not None:
            payload = {
                "episode": epi,
                "grasp_ok": int(det["grasp_ok"]),
                "grasp_and_lift_ok": int(det["lift_ok"]),
                "max_lift_norm": det["max_lift"],
            }
            if img_path is not None:
                payload["plot"] = wb.Image(img_path)
            wb.log(payload, step=epi)

    summary = dict(
        episodes_total=total,
        episodes_with_stable_grasp=ok_g,
        episodes_with_grasp_and_lift=ok_gl,
        ratio_grasp = ok_g  / max(total,1),
        ratio_grasp_and_lift = ok_gl / max(total,1),
        z_col=args.z_col,
        grip_col=grip_col,
        h_min_norm=args.h_min_norm,
        grip_close_thr=args.grip_close_thr,
        n_grasp=args.n_grasp,
        m_lift=args.m_lift,
    )
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[SUMMARY]")
    for k,v in summary.items():
        print(f"{k}: {v}")
    print(f"\nSaved plots to {plot_dir}")
    print(f"Saved summary.json to {args.out_dir}")

    if run is not None:
        # 同步到仪表盘的 Summary
        for k,v in summary.items():
            run.summary[k] = v
        run.finish()

if __name__ == "__main__":
    main()
