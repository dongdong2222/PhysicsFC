# tools/measure_trap_range.py 결과 (npz) 분석. 성공 = 트래핑 접촉 (success + window_fail).
# 다음 학습 분포에서 조절할 수 있는 샘플 값으로 셀을 나눠 셀별 성공률, 10 % 구간별 분포, X % 미만 셀 제외 효과,
# 공중볼 φ 최소값 규칙 (30 % 기준) 을 출력한다.
#   python tools/analyze_trap_range.py trap_range.npz [제외 기준 %, 기본 30]
#   공중볼 셀: 발사각 φ (5°) x 지정 착지 거리 fall_dist (0.5 m) x 공이 오는 쪽 (캐릭터 heading 기준 front |a|<45 / side / back |a|>135)
#   땅볼 셀:   도착 시간 (0.1 s) x 조준 거리 fall_dist (0.5 m) x 조준점 방향 (캐릭터 속도 방향 기준 front / side / back)
import sys
import numpy as np

G = 9.81
MIN_N = 100  # 이보다 샘플이 적은 셀은 판단하지 않음
def wrap_deg(a):
    return (np.degrees(a) + 180.0) % 360.0 - 180.0


def air_landing(d):
    # 공 중심이 반지름 높이로 돌아오는 시각과 위치. 연속 linear damping c: v' = -c v - g z
    c, r = float(d["lin_damp"]), float(d["ball_radius"])
    z0, vz0 = d["ball_z"], d["ball_vz"]
    def z(t):
        return z0 + (vz0 + G / c) / c * (1 - np.exp(-c * t)) - G * t / c
    lo = np.full_like(z0, 1e-3)
    hi = np.full_like(z0, 6.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        up = z(mid) > r
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    t = 0.5 * (lo + hi)
    f = (1 - np.exp(-c * t)) / c
    x = d["ball_x"] + d["ball_vx"] * f
    y = d["ball_y"] + d["ball_vy"] * f
    e = np.exp(-c * t)
    vx, vy = d["ball_vx"] * e, d["ball_vy"] * e
    vz = (vz0 + G / c) * e - G / c
    return t, x, y, np.sqrt(vx * vx + vy * vy + vz * vz), np.degrees(np.arctan2(-vz, np.hypot(vx, vy)))


def features(d):
    g = d["is_ground"] == 1
    er = d["end_reason"].astype(int)
    f = {"ground": g, "succ": (er == 1) | (er == 2), "end_reason": er}
    yaw = d["root_yaw"]
    rx, ry, vx, vy = d["root_x"], d["root_y"], d["root_vx"], d["root_vy"]

    t_land, lx, ly, v_land, descent = air_landing(d)
    # 공이 도착하는 지점 / 시각: 공중볼은 실제 (감쇠) 착지점, 땅볼은 조준점
    T = np.where(g, d["arrive_time"], t_land)
    px = np.where(g, d["fall_x"], lx)
    py = np.where(g, d["fall_y"], ly)
    dx, dy = px - rx, py - ry
    dist = np.hypot(dx, dy)
    f["T"] = T
    f["dist"] = dist
    f["req_speed"] = dist / T
    f["dir_to_point"] = wrap_deg(np.arctan2(dy, dx) - yaw)          # 0 = 정면, ±180 = 뒤
    # 공이 오는 쪽 (캐릭터 기준). 0 = 정면에서 날아옴, ±180 = 등 뒤에서
    f["ball_from"] = wrap_deg(d["launch_theta"] + np.pi - yaw)
    f["root_speed"] = np.hypot(vx, vy)
    u = np.stack([dx, dy], -1) / np.maximum(dist, 1e-6)[:, None]
    f["v_toward"] = vx * u[:, 0] + vy * u[:, 1]                     # 캐릭터 속도의 도착점 방향 성분
    # 지금 속도로 T 동안 가면 도착점에서 얼마나 벗어나나 (방향 전환 / 감속이 필요한 양)
    f["miss_if_coast"] = np.hypot(px - (rx + vx * T), py - (ry + vy * T))
    f["vel_dir"] = wrap_deg(np.arctan2(vy, vx) - yaw)

    # 샘플 값 (다음 학습 분포에서 직접 조절하는 값)
    f["speed"] = d["speed"]
    f["phi"] = np.degrees(d["phi"])
    f["arrive_speed"] = d["arrive_speed"]
    f["arrive_time"] = d["arrive_time"]
    f["fall_dist"] = d["fall_dist"]
    f["fall_theta_rel"] = wrap_deg(d["fall_theta"] - yaw)
    f["part"] = d["part"].astype(int)
    f["land_err"] = np.hypot(lx - d["fall_x"], ly - d["fall_y"])   # 공중볼: 지정 착지점과 실제 착지점 차이
    f["v_land"] = v_land
    f["descent"] = descent
    return f


def sector(a):
    a = np.abs(a)
    return np.where(a < 45, 0, np.where(a < 135, 1, 2))


SECTORS = ["front", "side", "back"]


def analyze(f, d, thr):
    g = f["ground"]
    names = d["end_reason_names"]
    f["fall_dir_relvel"] = wrap_deg(d["fall_theta"] - np.arctan2(d["root_vy"], d["root_vx"]))
    specs = {
        "air": (~g, "phi", np.arange(15, 60.01, 5), "fall_dist", np.arange(0, 4.01, 0.5), sector(f["ball_from"])),
        "ground": (g, "arrive_time", np.arange(1, 2.001, 0.1), "fall_dist", np.arange(0, 4.01, 0.5), sector(f["fall_dir_relvel"])),
    }
    for name, (m, xk, bx, yk, by, sec) in specs.items():
        s, er, N = f["succ"][m], f["end_reason"][m], int(m.sum())
        print(f"\n===== {name}: {N} episodes, 성공 (접촉) {s.mean():.3f}")
        print("  종료 사유", {names[i]: round(float((er == i + 1).mean()), 3) for i in range(len(names)) if (er == i + 1).any()})
        ix = np.clip(np.digitize(f[xk][m], bx) - 1, 0, len(bx) - 2)
        iy = np.clip(np.digitize(f[yk][m], by) - 1, 0, len(by) - 2)
        isec = sec[m]
        shape = [3, len(by) - 1, len(bx) - 1]
        count = np.zeros(shape, dtype=int)
        succ_sum = np.zeros(shape)
        np.add.at(count, (isec, iy, ix), 1)
        np.add.at(succ_sum, (isec, iy, ix), s)
        ok = count >= MIN_N
        rate = np.where(ok, succ_sum / np.maximum(count, 1), np.nan)
        for k in range(3):
            print(f"  [{SECTORS[k]}] 성공률 % (행 {yk}, 열 {xk}, '.' = 샘플 < {MIN_N})")
            print("            " + "".join(f"{v:>6g}" for v in bx[:-1]))
            for j in range(len(by) - 1):
                print(f"  {by[j]:3.1f}-{by[j + 1]:3.1f}  " + "".join("     ." if np.isnan(r) else f"{100 * r:6.0f}" for r in rate[k, j]))
        band = np.minimum(np.floor(np.nan_to_num(rate) * 10), 9)
        print("  성공률 구간    셀   분포 비율   구간 내 성공률")
        for b in range(10):
            c = ok & (band == b)
            n = count[c].sum()
            print(f"   {b * 10:3d}-{b * 10 + 10:3d}%  {c.sum():5d}   {n / N:8.3f}   {succ_sum[c].sum() / max(n, 1):10.3f}")
        print("  X% 미만 셀 제외   제외 비율   제외 후 성공률")
        for X in range(10, 100, 10):
            keep = ~ok | (rate >= X / 100)  # 샘플이 적은 셀은 남긴다
            n = count[keep].sum()
            print(f"   {X:3d}%            {1 - n / N:8.3f}   {succ_sum[keep].sum() / n:12.3f}")
        if name == "air":
            print(f"  공중볼 φ 최소값 ({thr}% 미만 셀 제외, 성공률이 φ 에 대해 단조 증가할 때만 규칙으로 표시)")
            for k in range(3):
                row = []
                for j in range(len(by) - 1):
                    r = rate[k, j]
                    if np.isnan(r).all():
                        row.append("   .")
                        continue
                    below = [i for i in range(len(r)) if not np.isnan(r[i]) and r[i] < thr / 100]
                    first_ok = len(below)  # 이 φ 구간부터 남긴다. 그 구간 샘플이 적으면 판단하지 않음
                    known = first_ok < len(r) and not np.isnan(r[first_ok])
                    row.append(f"{bx[first_ok]:4g}" if below == list(range(first_ok)) and known else "   ?")
                print(f"   {SECTORS[k]:5s} fall_dist " + " ".join(f"{by[j]:.1f}-{by[j + 1]:.1f}:{row[j]}" for j in range(len(by) - 1)))


if __name__ == "__main__":
    d = dict(np.load(sys.argv[1]))
    analyze(features(d), d, float(sys.argv[2]) if len(sys.argv) > 2 else 30.0)
