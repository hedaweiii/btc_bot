"""
BTC Active Range Bot

每 5 分钟执行一次；当信号变化时发送提醒。
每天北京时间 9:00 发送一次日报。

必需环境变量: FEISHU_WEBHOOK
可选环境变量: MODE -> conservador | equilibrado | agresivo
"""

import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

FEISHU_WEBHOOK = os.environ.get("FEISHU_WEBHOOK", "")
MODE = os.environ.get("MODE", "agresivo").lower()
STATE_FILE = Path("state.json")
BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"
BINANCE_TICK = "https://data-api.binance.vision/api/v3/ticker/price"

LEN_RANGE = 10
SIZE_FACTOR = 1.0
CTX_LEN = 5
CTX_THRESH = 0.5
DAILY_HOUR = 9
VOL_PERIOD = 20

TF_ORDER = ["15m", "1h", "4h", "1d", "1w"]
TF_LABELS = {"15m": "15M", "1h": "1H", "4h": "4H", "1d": "1D", "1w": "1W"}
TF_LIMITS = {"15m": 500, "1h": 500, "4h": 300, "1d": 200, "1w": 100}
TF_WEIGHTS = {"15m": 1, "1h": 2, "4h": 3, "1d": 4, "1w": 5}

# 模式配置：
#   thr           -> 强信号阈值（买入 / 卖出）
#   retest_tot    -> 出现 retest 确认时的最低分数
#   rsi_*         -> RSI 评分阈值
#   vol_*         -> 成交量确认阈值
#   mtf_*         -> 多周期一致性阈值
#   sl_atr/tp_atr -> 止损止盈 ATR 倍数
#   allow_weak    -> 是否允许弱信号
MODES = {
    "conservador": {
        "label": "保守",
        "thr": 7,
        "retest_tot": 5,
        "rsi_oversold": 35,
        "rsi_buy": 42,
        "rsi_neutral_hi": 58,
        "rsi_sell": 72,
        "rsi_overbought": 78,
        "vol_strong": 1.8,
        "vol_good": 1.3,
        "vol_weak": 0.8,
        "mtf_strong": 0.85,
        "mtf_medium": 0.65,
        "sl_atr": 1.5,
        "tp_atr": 2.5,
        "allow_weak": False,
    },
    "equilibrado": {
        "label": "均衡",
        "thr": 5,
        "retest_tot": 3,
        "rsi_oversold": 30,
        "rsi_buy": 45,
        "rsi_neutral_hi": 55,
        "rsi_sell": 55,
        "rsi_overbought": 80,
        "vol_strong": 1.5,
        "vol_good": 1.2,
        "vol_weak": 0.8,
        "mtf_strong": 0.75,
        "mtf_medium": 0.50,
        "sl_atr": 1.0,
        "tp_atr": 1.5,
        "allow_weak": True,
    },
    "agresivo": {
        "label": "激进",
        "thr": 3,
        "retest_tot": 1,
        "rsi_oversold": 38,
        "rsi_buy": 55,
        "rsi_neutral_hi": 52,
        "rsi_sell": 50,
        "rsi_overbought": 75,
        "vol_strong": 1.2,
        "vol_good": 1.0,
        "vol_weak": 0.7,
        "mtf_strong": 0.50,
        "mtf_medium": 0.30,
        "sl_atr": 0.7,
        "tp_atr": 1.0,
        "allow_weak": True,
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def get_cfg():
    """返回当前模式配置，未知模式时回退到 equilibrado。"""
    if MODE not in MODES:
        log.warning(f"未知模式 '{MODE}'，已切换到 'equilibrado'。")
        return MODES["equilibrado"]
    return MODES[MODE]


def load_state():
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
            if "signals" in data:
                if data.get("mode") != MODE:
                    log.info(f"模式已变更（{data.get('mode')} -> {MODE}），重置历史信号。")
                    return {
                        "signals": {tf: None for tf in TF_ORDER},
                        "last_daily": None,
                        "mode": MODE,
                    }
                for tf in TF_ORDER:
                    data["signals"].setdefault(tf, None)
                return data
            return {"signals": {tf: None for tf in TF_ORDER}, "last_daily": None, "mode": MODE}
        except Exception:
            pass
    return {"signals": {tf: None for tf in TF_ORDER}, "last_daily": None, "mode": MODE}


def save_state(state):
    state["mode"] = MODE
    STATE_FILE.write_text(json.dumps(state, indent=2))


def now_beijing():
    return datetime.now(timezone.utc) + timedelta(hours=8)


def calc_rsi(closes, n=14):
    if len(closes) < n + 1:
        return [None] * len(closes)
    rsi = [None] * n
    gains = [max(closes[i] - closes[i - 1], 0) for i in range(1, n + 1)]
    losses = [max(closes[i - 1] - closes[i], 0) for i in range(1, n + 1)]
    avg_gain, avg_loss = sum(gains) / n, sum(losses) / n
    rsi.append(100 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss))
    for i in range(n + 1, len(closes)):
        delta = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (n - 1) + max(delta, 0)) / n
        avg_loss = (avg_loss * (n - 1) + max(-delta, 0)) / n
        rsi.append(100 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss))
    return rsi


def calc_atr(candles, n=14):
    if len(candles) < n + 1:
        return None
    recent = candles[-(n + 1) :]
    return sum(
        max(
            recent[i]["high"] - recent[i]["low"],
            abs(recent[i]["high"] - recent[i - 1]["close"]),
            abs(recent[i]["low"] - recent[i - 1]["close"]),
        )
        for i in range(1, n + 1)
    ) / n


def calc_volume_ratio(candles, period=VOL_PERIOD):
    if len(candles) < period + 1:
        return None
    volumes = [c["volume"] for c in candles]
    avg_volume = sum(volumes[-(period + 1) : -1]) / period
    if avg_volume == 0:
        return None
    return volumes[-1] / avg_volume


def is_big_candle(candles, idx, length, factor):
    if idx < length:
        return False
    slice_candles = candles[idx - length : idx]
    range_avg = sum(c["high"] - c["low"] for c in slice_candles) / length
    body_avg = sum(abs(c["close"] - c["open"]) for c in slice_candles) / length
    candle = candles[idx]
    return (candle["high"] - candle["low"] > range_avg * factor) or (
        abs(candle["close"] - candle["open"]) > body_avg * factor
    )


def is_continuation(candles, idx, ctx_len, thresh):
    if idx < ctx_len:
        return False
    current = candles[idx]
    bulls = bears = 0
    for i in range(1, ctx_len + 1):
        candle = candles[idx - i]
        if candle["close"] > candle["open"]:
            bulls += 1
        if candle["close"] < candle["open"]:
            bears += 1
    return ((current["close"] > current["open"]) and bulls / ctx_len >= thresh) or (
        (current["close"] < current["open"]) and bears / ctx_len >= thresh
    )


def detect_active_range(candles, length, factor):
    in_range = False
    high = low = None
    hit_high = hit_low = False
    start_bar = None
    phase = 0
    touches_low = touches_high = 0
    went_below = went_above = False
    retest_buy = retest_sell = False
    below_bar = above_bar = None
    allow_from_low = allow_from_high = False
    in_touch_low = in_touch_high = False
    is_cont = False
    states = []

    for i, candle in enumerate(candles):
        did_break = False
        if in_range and high is not None and (candle["close"] > high or candle["close"] < low):
            in_range = False
            high = low = None
            hit_high = hit_low = False
            start_bar = None
            phase = 0
            touches_low = touches_high = 0
            went_below = went_above = False
            retest_buy = retest_sell = False
            below_bar = above_bar = None
            allow_from_low = allow_from_high = False
            in_touch_low = in_touch_high = False
            is_cont = False
            did_break = True

        big = is_big_candle(candles, i, length, factor)
        if (did_break or not in_range) and big:
            high = candle["high"]
            low = candle["low"]
            in_range = True
            hit_high = hit_low = False
            start_bar = i
            phase = 0
            touches_low = touches_high = 0
            went_below = went_above = False
            retest_buy = retest_sell = False
            below_bar = above_bar = None
            allow_from_low = allow_from_high = False
            in_touch_low = in_touch_high = False
            is_cont = is_continuation(candles, i, CTX_LEN, CTX_THRESH)

        if in_range and start_bar is not None and i > start_bar:
            rng = high - low
            if candle["close"] > low + rng * 0.25:
                allow_from_low = True
            if candle["close"] < low + rng * 0.75:
                allow_from_high = True

            if candle["low"] <= low and candle["close"] >= low:
                if not in_touch_low:
                    if allow_from_low or touches_low == 0:
                        touches_low += 1
                        allow_from_low = False
                        if not went_below:
                            went_below = True
                            below_bar = i
                        in_touch_low = True
                hit_low = True
            else:
                in_touch_low = False

            if candle["high"] >= high and candle["close"] <= high:
                if not in_touch_high:
                    if allow_from_high or touches_high == 0:
                        touches_high += 1
                        allow_from_high = False
                        if not went_above:
                            went_above = True
                            above_bar = i
                        in_touch_high = True
                hit_high = True
            else:
                in_touch_high = False

            if went_below and phase == 0:
                phase = 1
            if went_above and phase == 0:
                phase = 2
            if went_above and phase == 1:
                phase = 2
                retest_buy = False
            if went_below and phase == 2:
                phase = 1
                retest_sell = False
            if hit_high and hit_low and phase != 2:
                phase = 2
            if went_below and not retest_buy and below_bar is not None and i > below_bar and candle["low"] <= low:
                retest_buy = True
            if went_above and not retest_sell and above_bar is not None and i > above_bar and candle["high"] >= high:
                retest_sell = True

        states.append(
            {
                "in": in_range,
                "hi": high,
                "lo": low,
                "hit_hi": hit_high,
                "hit_lo": hit_low,
                "phase": phase,
                "is_cont": is_cont,
                "touches_lo": touches_low,
                "touches_hi": touches_high,
                "went_below": went_below,
                "went_above": went_above,
                "retest_buy": retest_buy,
                "retest_sell": retest_sell,
            }
        )
    return states


def calc_bias(candles):
    last = detect_active_range(candles, LEN_RANGE, SIZE_FACTOR)[-1]
    if not last or not last["in"]:
        return {"bias": 0}
    return {"bias": 1 if last["phase"] == 1 else (-1 if last["phase"] == 2 else 0)}


def fetch_current_price():
    try:
        response = requests.get(BINANCE_TICK, params={"symbol": "BTCUSDT"}, timeout=10)
        response.raise_for_status()
        return float(response.json()["price"])
    except Exception as e:
        log.warning(f"获取当前价格失败: {e}")
        return None


def compute_signal(candles_map, tf_key, cfg, current_price=None):
    """根据当前模式配置，计算指定周期的信号。"""
    candles = candles_map.get(tf_key)
    if not candles or len(candles) < LEN_RANGE + 5:
        return None

    closes = [c["close"] for c in candles]
    price = current_price if current_price is not None else closes[-1]
    states = detect_active_range(candles, LEN_RANGE, SIZE_FACTOR)
    last = states[-1]
    rsi = calc_rsi(closes, 14)[-1]
    atr = calc_atr(candles, 14)
    vol_ratio = calc_volume_ratio(candles)
    score_items = []
    total_score = 0

    def add(name, score, max_score, label, color):
        nonlocal total_score
        score_items.append(
            {"name": name, "score": score, "max": max_score, "label": label, "color": color}
        )
        total_score += score

    if last["in"] and last["hi"]:
        if last["phase"] == 1 and last["retest_buy"]:
            add("Zone", 3, 3, "买入 retest 已确认", "green")
        elif last["phase"] == 1 and last["went_below"]:
            add("Zone", 2, 3, "触及下沿区域", "green")
        elif last["phase"] == 1:
            add("Zone", 1, 3, "偏多阶段", "green")
        elif last["phase"] == 2 and last["retest_sell"]:
            add("Zone", -3, 3, "卖出 retest 已确认", "red")
        elif last["phase"] == 2 and last["went_above"]:
            add("Zone", -2, 3, "触及上沿区域", "red")
        elif last["phase"] == 2:
            add("Zone", -1, 3, "偏空阶段", "red")
        else:
            add("Zone", 0, 3, "无明确阶段", "yellow")
    else:
        add("Zone", 0, 3, "无活跃区间", "yellow")

    if rsi is not None:
        if rsi < cfg["rsi_oversold"]:
            add("RSI", 3, 3, f"RSI {rsi:.1f}，极度超卖", "green")
        elif rsi < cfg["rsi_buy"]:
            add("RSI", 2, 3, f"RSI {rsi:.1f}，买盘占优", "green")
        elif rsi < cfg["rsi_neutral_hi"]:
            add("RSI", 0, 3, f"RSI {rsi:.1f}，中性", "yellow")
        elif rsi < cfg["rsi_sell"]:
            add("RSI", -1, 3, f"RSI {rsi:.1f}，谨慎", "yellow")
        elif rsi < cfg["rsi_overbought"]:
            add("RSI", -2, 3, f"RSI {rsi:.1f}，超买", "red")
        else:
            add("RSI", -3, 3, f"RSI {rsi:.1f}，极度超买", "red")

    if vol_ratio is not None:
        cur_bias = 1 if last["phase"] == 1 else (-1 if last["phase"] == 2 else 0)
        if vol_ratio >= cfg["vol_strong"]:
            if cur_bias != 0:
                add("Vol", 2, 2, f"成交量 {vol_ratio:.1f}x，确认当前方向", "green")
            else:
                add("Vol", 1, 2, f"成交量 {vol_ratio:.1f}x，但方向不明", "yellow")
        elif vol_ratio >= cfg["vol_good"]:
            add("Vol", 1, 2, f"成交量 {vol_ratio:.1f}x，高于均值", "green")
        elif vol_ratio < cfg["vol_weak"] and last["in"]:
            add("Vol", -1, 2, f"成交量 {vol_ratio:.1f}x，信号偏弱", "red")
        else:
            add("Vol", 0, 2, f"成交量 {vol_ratio:.1f}x，正常", "yellow")

    biases = [calc_bias(v) for k, v in candles_map.items() if k in TF_LABELS]
    cur_bias = 1 if last["phase"] == 1 else (-1 if last["phase"] == 2 else 0)
    active_biases = [b for b in biases if b["bias"] != 0]
    aligned = sum(1 for b in active_biases if b["bias"] == cur_bias)
    if active_biases:
        ratio = aligned / len(active_biases)
        if ratio >= cfg["mtf_strong"]:
            add("MTF", 2, 2, f"{aligned}/{len(active_biases)} 个周期方向一致", "green")
        elif ratio >= cfg["mtf_medium"]:
            add("MTF", 1, 2, "多周期部分一致", "yellow")
        elif ratio == 0:
            add("MTF", -2, 2, "多周期分歧明显", "red")
        else:
            add("MTF", -1, 2, "多周期一致性较弱", "yellow")

    if last["in"]:
        add("Context", 1 if last["is_cont"] else -1, 1, "趋势延续" if last["is_cont"] else "可能反转", "green")
        touches = last["touches_lo"] if last["phase"] == 1 else last["touches_hi"]
        if touches >= 3:
            add("Strength", 2, 2, f"区域已测试 {touches} 次", "green")
        elif touches == 2:
            add("Strength", 1, 2, "区域 2 次确认", "green")
        else:
            add("Strength", 0, 2, "区域尚未充分确认", "yellow")

    zone_score = next((x["score"] for x in score_items if x["name"] == "Zone"), 0)
    max_score = sum(x["max"] for x in score_items)
    threshold = cfg["thr"]
    weak_threshold = math.ceil(threshold / 2)
    retest_buy = last["in"] and last["phase"] == 1 and last["retest_buy"]
    retest_sell = last["in"] and last["phase"] == 2 and last["retest_sell"]

    if retest_buy and total_score >= cfg["retest_tot"] and zone_score > 0:
        action, css = "买入", "buy"
    elif total_score >= threshold and zone_score > 0:
        action, css = "买入", "buy"
    elif cfg["allow_weak"] and total_score >= weak_threshold and zone_score > 0:
        action, css = "弱买入", "buy"
    elif retest_sell and total_score <= -cfg["retest_tot"] and zone_score < 0:
        action, css = "卖出", "sell"
    elif total_score <= -threshold and zone_score < 0:
        action, css = "卖出", "sell"
    elif cfg["allow_weak"] and total_score <= -weak_threshold and zone_score < 0:
        action, css = "弱卖出", "sell"
    elif not last["in"]:
        action, css = "无区间", "wait"
    else:
        action, css = "观望", "neutral"

    return {
        "action": action,
        "css": css,
        "tot": total_score,
        "max_s": max_score,
        "price": price,
        "rsi": rsi,
        "atr": atr,
        "vol_ratio": vol_ratio,
        "range_state": last,
    }


def calc_probability(sigs):
    if not sigs:
        return 50, "neutral", "无数据", "■■■■■"

    n_buy = sum(1 for s in sigs.values() if s["css"] == "buy")
    n_sell = sum(1 for s in sigs.values() if s["css"] == "sell")
    if n_buy == 0 and n_sell == 0:
        return 50, "neutral", "方向不明确", "■■■■■"

    direction = "buy" if n_buy >= n_sell else "sell"
    total_weight = 0
    weighted_score = 0
    for tf, signal in sigs.items():
        weight = TF_WEIGHTS.get(tf, 1)
        total_weight += weight
        max_score = signal["max_s"] if signal["max_s"] > 0 else 1
        norm = (signal["tot"] + max_score) / (max_score * 2) * 100
        if direction == "buy" and signal["css"] == "sell":
            norm = 100 - norm
        if direction == "sell" and signal["css"] == "buy":
            norm = 100 - norm
        weighted_score += norm * weight

    prob = weighted_score / total_weight if total_weight > 0 else 50
    vol_adj = 0
    for signal in sigs.values():
        vol_ratio = signal.get("vol_ratio")
        if vol_ratio is None:
            continue
        if vol_ratio >= 1.5:
            vol_adj += 2
        elif vol_ratio >= 1.2:
            vol_adj += 1
        elif vol_ratio < 0.8:
            vol_adj -= 2
    prob += max(-6, min(6, vol_adj))

    rsi_adj = 0
    for signal in sigs.values():
        rsi = signal.get("rsi")
        if rsi is None:
            continue
        if direction == "buy":
            if rsi < 30:
                rsi_adj += 2
            elif rsi < 45:
                rsi_adj += 1
            elif rsi > 70:
                rsi_adj -= 2
        else:
            if rsi > 70:
                rsi_adj += 2
            elif rsi > 55:
                rsi_adj += 1
            elif rsi < 30:
                rsi_adj -= 2
    prob += max(-4, min(4, rsi_adj))

    for signal in sigs.values():
        range_state = signal.get("range_state", {})
        if direction == "buy" and range_state.get("retest_buy"):
            prob += 5
            break
        if direction == "sell" and range_state.get("retest_sell"):
            prob += 5
            break

    if n_buy > 0 and n_sell > 0:
        prob -= min(n_buy, n_sell) * 3

    prob = max(10, min(95, round(prob)))
    if prob >= 80:
        label = "很高"
    elif prob >= 65:
        label = "较高"
    elif prob >= 50:
        label = "中等"
    elif prob >= 35:
        label = "较低"
    else:
        label = "很低"
    bar = "■" * round(prob / 10) + "□" * (10 - round(prob / 10))
    return prob, direction, label, bar


def prob_header(sigs):
    prob, direction, label, bar = calc_probability(sigs)
    emoji = "🟢" if prob >= 80 else ("🟡" if prob >= 65 else ("🟠" if prob >= 50 else "🔴"))
    direction_text = "买入" if direction == "buy" else ("卖出" if direction == "sell" else "中性")
    return f"{emoji} 胜率估计: {prob}%\n{bar} {label}\n主导方向: {direction_text}"


def calc_confluence(sigs):
    n_buy = n_sell = n_wait = n_none = 0
    for signal in sigs.values():
        css = signal["css"]
        if css == "buy":
            n_buy += 1
        elif css == "sell":
            n_sell += 1
        elif css == "wait":
            n_none += 1
        else:
            n_wait += 1

    total = len(sigs)
    if n_buy >= 4:
        return "🟢🟢", f"强多头，{n_buy}/{total} 个周期为买入", n_buy, n_sell, n_wait, n_none
    if n_buy >= 3:
        return "🟢", f"偏多，{n_buy}/{total} 个周期为买入", n_buy, n_sell, n_wait, n_none
    if n_sell >= 4:
        return "🔴🔴", f"强空头，{n_sell}/{total} 个周期为卖出", n_buy, n_sell, n_wait, n_none
    if n_sell >= 3:
        return "🔴", f"偏空，{n_sell}/{total} 个周期为卖出", n_buy, n_sell, n_wait, n_none
    if n_buy > 0 and n_sell > 0:
        return "⚖️", f"多空分歧，买入 {n_buy} vs 卖出 {n_sell}", n_buy, n_sell, n_wait, n_none
    if n_wait >= 3:
        return "🟡", f"方向不清晰，{n_wait}/{total} 个周期在观望", n_buy, n_sell, n_wait, n_none
    return "⚪", "大多数周期暂无活跃区间", n_buy, n_sell, n_wait, n_none


def fetch_all_candles():
    data = {}
    for tf in TF_ORDER:
        response = requests.get(
            BINANCE_URL,
            params={"symbol": "BTCUSDT", "interval": tf, "limit": TF_LIMITS[tf]},
            timeout=15,
        )
        response.raise_for_status()
        data[tf] = [
            {
                "ts": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            }
            for k in response.json()[:-1]
        ]
        log.info(f"{TF_LABELS[tf]}: 已获取 {len(data[tf])} 根已收盘 K 线")
    return data


def feishu_send(text):
    response = requests.post(
        FEISHU_WEBHOOK,
        json={"msg_type": "text", "content": {"text": text}},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("code") == 0


def fmt_price(price):
    return f"{price:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def vol_emoji(vol_ratio):
    if vol_ratio is None:
        return ""
    if vol_ratio >= 1.5:
        return " 🔥"
    if vol_ratio >= 1.2:
        return " ⬆️"
    if vol_ratio < 0.8:
        return " ⬇️"
    return ""


def build_tf_block(tf, signal, cfg):
    emoji = {"buy": "🟢", "sell": "🔴", "wait": "⚪", "neutral": "🟡"}.get(signal["css"], "🟡")
    rsi_txt = f" | RSI {signal['rsi']:.0f}" if signal["rsi"] is not None else ""
    rng_txt = " | 活跃区间" if signal["range_state"]["in"] else ""
    vol_txt = f" | 量能 {signal['vol_ratio']:.1f}x{vol_emoji(signal['vol_ratio'])}" if signal.get("vol_ratio") else ""
    price_fmt = fmt_price(signal["price"])

    if signal["atr"] and signal["css"] in ("buy", "sell"):
        is_buy = signal["css"] == "buy"
        sl = signal["price"] - signal["atr"] * cfg["sl_atr"] if is_buy else signal["price"] + signal["atr"] * cfg["sl_atr"]
        tp = signal["price"] + signal["atr"] * cfg["tp_atr"] if is_buy else signal["price"] - signal["atr"] * cfg["tp_atr"]
        sl_pct = abs(sl - signal["price"]) / signal["price"] * 100
        tp_pct = abs(tp - signal["price"]) / signal["price"] * 100
        rr = cfg["tp_atr"] / cfg["sl_atr"]
        sl_sign = "-" if is_buy else "+"
        tp_sign = "+" if is_buy else "-"
        levels = (
            f"   止损: ${round(sl):,} ({sl_sign}{sl_pct:.2f}%)\n"
            f"   止盈: ${round(tp):,} ({tp_sign}{tp_pct:.2f}%)\n"
            f"   盈亏比: {rr:.1f}:1 | SL×{cfg['sl_atr']} ATR | TP×{cfg['tp_atr']} ATR"
        ).replace(",", ".")
    else:
        levels = "   当前无止损止盈建议"

    range_state = signal["range_state"]
    range_line = ""
    if range_state.get("in") and range_state.get("hi"):
        range_line = f"   区间: ${round(range_state['lo']):,} - ${round(range_state['hi']):,}\n".replace(",", ".")

    return (
        f"{emoji} {TF_LABELS[tf]}: {signal['action']}{rsi_txt}{vol_txt}{rng_txt}\n"
        f"   价格: ${price_fmt}\n"
        f"{range_line}{levels}"
    )


def build_alert_message(sigs, changed_tfs, cfg):
    timestamp = now_beijing().strftime("%d/%m %H:%M")
    changed = ", ".join(TF_LABELS[t] for t in TF_ORDER if t in changed_tfs)
    conf_emoji, conf_text, n_buy, n_sell, n_wait, n_none = calc_confluence(sigs)
    blocks = [build_tf_block(tf, sigs[tf], cfg) for tf in TF_ORDER if sigs.get(tf)]
    divider = "\n" + ("─" * 26) + "\n"
    return (
        f"BTC Active Range 信号变更\n"
        f"模式: {cfg['label']} | 变更周期: {changed}"
        f"{divider}"
        f"{prob_header(sigs)}"
        f"{divider}"
        f"{conf_emoji} 共振情况: {conf_text}\n"
        f"统计: 买入 {n_buy} | 卖出 {n_sell} | 观望 {n_wait} | 无区间 {n_none}"
        f"{divider}"
        + divider.join(blocks)
        + f"{divider}北京时间: {timestamp}"
    )


def build_daily_message(sigs, cfg):
    date_text = now_beijing().strftime("%d/%m/%Y")
    conf_emoji, conf_text, n_buy, n_sell, n_wait, n_none = calc_confluence(sigs)
    blocks = [build_tf_block(tf, sigs[tf], cfg) for tf in TF_ORDER if sigs.get(tf)]
    divider = "\n" + ("─" * 26) + "\n"
    return (
        f"BTC Active Range 每日报告 | {date_text}\n"
        f"模式: {cfg['label']} | 北京时间 09:00"
        f"{divider}"
        f"{prob_header(sigs)}"
        f"{divider}"
        f"{conf_emoji} 共振情况: {conf_text}\n"
        f"统计: 买入 {n_buy} | 卖出 {n_sell} | 观望 {n_wait} | 无区间 {n_none}"
        f"{divider}"
        + divider.join(blocks)
        + f"{divider}下次日报: 明天北京时间 09:00"
    )


def main():
    log.info("BTC Active Range Bot 启动")
    if not FEISHU_WEBHOOK:
        log.error("缺少 FEISHU_WEBHOOK 环境变量")
        return

    cfg = get_cfg()
    log.info(f"当前模式: {cfg['label']} (阈值={cfg['thr']}, 弱信号={cfg['allow_weak']})")

    state = load_state()
    last_signals = state["signals"]
    last_daily = state.get("last_daily")
    log.info(f"历史信号: {last_signals}")

    log.info("开始下载 Binance K 线（仅已收盘 K 线）...")
    try:
        candles_map = fetch_all_candles()
    except Exception as e:
        log.error(f"Binance 数据获取失败: {e}")
        return

    log.info("获取当前价格...")
    current_price = fetch_current_price()
    if current_price:
        log.info(f"当前价格: ${current_price:,.2f}")
    else:
        log.warning("改用最近一根已收盘 K 线的收盘价")

    log.info("开始计算信号...")
    sigs = {}
    for tf in TF_ORDER:
        try:
            signal = compute_signal(candles_map, tf, cfg, current_price)
            if signal:
                sigs[tf] = signal
                vol_info = f"  Vol:{signal['vol_ratio']:.2f}x" if signal.get("vol_ratio") else ""
                log.info(f"{TF_LABELS[tf]}: {signal['action']}  score={signal['tot']:+d}  RSI={signal['rsi']:.1f}{vol_info}")
        except Exception as e:
            log.error(f"计算 {TF_LABELS[tf]} 信号失败: {e}")

    prob, direction, label, _ = calc_probability(sigs)
    log.info(f"综合胜率: {prob}% ({label})，主导方向={direction}")

    changed = []
    for tf in TF_ORDER:
        if tf in sigs:
            prev = last_signals.get(tf)
            curr = sigs[tf]["action"]
            if curr != prev:
                changed.append(tf)
                log.info(f"信号变化 {TF_LABELS[tf]}: {prev} -> {curr}")
            else:
                log.info(f"信号未变 {TF_LABELS[tf]}: {curr}")

    now_bj = now_beijing()
    today_str = now_bj.strftime("%Y-%m-%d")
    sent_daily = False

    if now_bj.hour == DAILY_HOUR and last_daily != today_str:
        log.info("发送每日报告...")
        if feishu_send(build_daily_message(sigs, cfg)):
            state["last_daily"] = today_str
            sent_daily = True
            log.info("每日报告发送成功。")
        else:
            log.error("每日报告发送失败。")

    if not changed:
        log.info("所有周期都没有变化，本次结束。")
        if sent_daily:
            save_state(state)
        return

    log.info(f"检测到变更周期: {[TF_LABELS[t] for t in changed]}")
    if feishu_send(build_alert_message(sigs, changed, cfg)):
        for tf in changed:
            state["signals"][tf] = sigs[tf]["action"]
        save_state(state)
        log.info("飞书提醒发送成功，状态已保存。")
    else:
        log.error("飞书提醒发送失败。")
        if sent_daily:
            save_state(state)


if __name__ == "__main__":
    main()
