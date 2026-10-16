"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Area, AreaChart,
} from "recharts";
import {
  Activity, TrendingUp, TrendingDown, Minus, Wifi, WifiOff, ShieldAlert,
  Wallet, ArrowUpRight, ArrowDownRight, AlertTriangle, ShieldCheck
} from "lucide-react";

// ── Types ────────────────────────────────────────────────────────────────────
interface PriceTick {
  price_usdt: number;
  price_usd: number;
  usdt_usd_rate: number;
  timestamp: number;
  time: string;
}

interface Prediction {
  symbol: string;
  timestamp: string;
  horizon: string;
  price: number;
  probability_down: number;
  probability_neutral: number;
  probability_up: number;
  predicted_class: string;
  confidence: number;
  model_version: string;
}

interface Alert {
  id: number;
  timestamp: string;
  action: string;
  entry_price: number;
  take_profit: number;
  stop_loss: number;
  confidence: number;
  status: string;
}

interface Trade {
  id?: number;
  side: string;
  entry_price: number;
  exit_price: number;
  quantity_btc: number;
  position_usd?: number;
  pnl_usd: number;
  pnl_pct: number;
  reason?: string;
  exit_reason?: string;
  opened_at?: string;
  closed_at?: string;
  balance_after?: number;
}

interface Portfolio {
  mode: string;
  initial_capital_usd: number;
  balance_usd: number;
  peak_balance_usd?: number;
  total_pnl_usd: number;
  total_return_pct: number;
  total_trades: number;
  winning_trades: number;
  win_rate_pct: number;
  profit_factor: number;
  max_drawdown_pct: number;
  open_position?: {
    side: string;
    entry_price: number;
    quantity_btc: number;
    usd_value: number;
    stop_loss_price: number;
    opened_at: string;
    unrealized_pnl_usd: number;
    unrealized_pnl_pct: number;
  } | null;
  recent_trades: Trade[];
}

// ── Dynamic URL Helpers (works for localhost, GitHub Pages, Netlify, Cloudflare, Vercel, and cloud) ────
const isExternalHost = (host: string) => {
  return (
    host.includes("workers.dev") ||
    host.includes("pages.dev") ||
    host.includes("vercel.app") ||
    host.includes("vercel.sh") ||
    host.includes("github.io") ||
    host.includes("netlify.app") ||
    host.includes("onrender.com")
  );
};

const getApiBase = () => {
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  if (typeof window !== "undefined") {
    if (window.location.port === "3000") {
      return `http://${window.location.hostname}:8000`;
    }
    if (isExternalHost(window.location.hostname)) {
      return "https://soumyadeep-04-btc-trader.hf.space";
    }
    return window.location.origin;
  }
  return "https://soumyadeep-04-btc-trader.hf.space";
};

const getWsUrl = () => {
  if (process.env.NEXT_PUBLIC_WS_URL) {
    return process.env.NEXT_PUBLIC_WS_URL;
  }
  if (typeof window !== "undefined") {
    if (window.location.port === "3000") {
      return `ws://${window.location.hostname}:8000/ws/price`;
    }
    if (isExternalHost(window.location.hostname)) {
      return "wss://soumyadeep-04-btc-trader.hf.space/ws/price";
    }
    const isHttps = window.location.protocol === "https:";
    const protocol = isHttps ? "wss:" : "ws:";
    return `${protocol}//${window.location.host}/ws/price`;
  }
  return "wss://soumyadeep-04-btc-trader.hf.space/ws/price";
};

// ── Constants ────────────────────────────────────────────────────────────────
const MAX_TICKS = 120; // keep last 120 ticks in chart (~2 min of trade data)

const DEMO_PREDICTION: Prediction = {
  symbol: "BTCUSDT",
  timestamp: new Date().toISOString(),
  horizon: "1h",
  price: 64230.5,
  probability_down: 0.15,
  probability_neutral: 0.55,
  probability_up: 0.30,
  predicted_class: "NEUTRAL",
  confidence: 0.55,
  model_version: "v2",
};

// ── Component ────────────────────────────────────────────────────────────────
export default function Dashboard() {
  const [livePriceUsdt, setLivePriceUsdt] = useState<number | null>(null);
  const [livePriceUsd, setLivePriceUsd] = useState<number | null>(null);
  const [prevPriceUsd, setPrevPriceUsd] = useState<number | null>(null);
  const [usdtRate, setUsdtRate] = useState<number>(1.0);
  const [ticks, setTicks] = useState<PriceTick[]>([]);
  const [wsStatus, setWsStatus] = useState<"connecting" | "live" | "disconnected">("connecting");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [botAlert, setBotAlert] = useState<Alert | null>(null);
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [isEmergencyStopping, setIsEmergencyStopping] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);


  const BINANCE_DIRECT_WS = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade";

  const connectWs = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    setWsStatus("connecting");

    // In cloud deployments (Cloudflare Workers, Vercel, Netlify), connect directly to Binance
    // for sub-second zero-proxy real-time ticks. In local dev, use local backend.
    const targetWsUrl = (typeof window !== "undefined" && isExternalHost(window.location.hostname))
      ? BINANCE_DIRECT_WS
      : getWsUrl();

    const ws = new WebSocket(targetWsUrl);
    wsRef.current = ws;

    ws.onopen = () => setWsStatus("live");

    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        let priceUsdt: number;
        let priceUsd: number;
        let rate = usdtRate;
        let tsMs: number;

        if (data.p) {
          // Direct Binance aggTrade format: { p: price, T: timestampMs, ... }
          priceUsdt = parseFloat(data.p);
          priceUsd = priceUsdt * (rate || 1.0);
          tsMs = typeof data.T === "number" ? data.T : Date.now();
        } else {
          // Backend proxy format
          priceUsdt = parseFloat(data.price_usdt);
          priceUsd  = parseFloat(data.price_usd);
          rate      = parseFloat(data.usdt_usd_rate) || rate;
          tsMs = typeof data.timestamp === "number" ? data.timestamp : new Date(data.timestamp || Date.now()).getTime();
        }

        const timeLabel = new Date(tsMs).toLocaleTimeString("en-IN", { hour12: false });

        setLivePriceUsdt(priceUsdt);
        setLivePriceUsd((prev) => { setPrevPriceUsd(prev); return priceUsd; });
        setUsdtRate(rate);
        setTicks((prev) => {
          const next = [...prev, { price_usdt: priceUsdt, price_usd: priceUsd, usdt_usd_rate: rate, timestamp: tsMs, time: timeLabel }];
          return next.slice(-MAX_TICKS);
        });
      } catch (e) {
        console.error("WS tick parse error:", e);
      }
    };

    ws.onclose = () => {
      setWsStatus("disconnected");
      reconnectTimer.current = setTimeout(connectWs, 2500);
    };

    ws.onerror = () => ws.close();
  }, [usdtRate]);

  useEffect(() => {
    connectWs();
    return () => {
      wsRef.current?.close();
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    };
  }, [connectWs]);

  
  const loadPortfolio = useCallback(async () => {
    try {
      const res = await fetch(`${getApiBase()}/api/v1/portfolio`);
      if (res.ok) {
        const data = await res.json();
        setPortfolio(data);
      }
    } catch (err) {
      console.error("Failed to load portfolio", err);
    }
  }, []);

  useEffect(() => {
    const loadPrediction = async () => {
      try {
        const res = await fetch(`${getApiBase()}/api/v1/prediction/latest`);
        if (res.ok) {
          const data = await res.json();
          setPrediction(data);
        } else {
          setPrediction(DEMO_PREDICTION);
        }
      } catch {
        setPrediction(DEMO_PREDICTION);
      }
    };
    
    const loadAlert = async () => {
      try {
        const res = await fetch(`${getApiBase()}/api/v1/alerts/latest`);
        if (res.ok) {
          const data = await res.json();
          setBotAlert(data);
        }
      } catch (err) {
        console.error("Failed to load alert", err);
      }
    };
    
    loadPrediction();
    loadAlert();
    loadPortfolio();

    const interval = setInterval(() => {
      loadPrediction();
      loadAlert();
      loadPortfolio();
    }, 15 * 1000); // 15 sec refresh
    return () => clearInterval(interval);
  }, [loadPortfolio]);

  // ── Emergency Kill Switch ─────────────────────────────────────────────────
  const handleEmergencyStop = async () => {
    if (!confirm("Are you sure you want to trigger the Emergency Kill-Switch and withdraw all open positions immediately?")) {
      return;
    }
    setIsEmergencyStopping(true);
    try {
      const res = await fetch(`${getApiBase()}/api/v1/trading/emergency-stop`, { method: "POST" });
      const data = await res.json();
      window.alert(`Emergency response: ${data.status || JSON.stringify(data)}`);
      await loadPortfolio();
    } catch (err) {
      window.alert("Failed to execute emergency stop: " + err);
    } finally {
      setIsEmergencyStopping(false);
    }
  };

  // ── Derived ───────────────────────────────────────────────────────────────
  const priceUp = livePriceUsd !== null && prevPriceUsd !== null && livePriceUsd >= prevPriceUsd;
  const priceChanged = livePriceUsd !== null && prevPriceUsd !== null && livePriceUsd !== prevPriceUsd;

  const isUp = prediction?.predicted_class === "UP";
  const isDown = prediction?.predicted_class === "DOWN";

  const domainMin = ticks.length > 1
    ? Math.min(...ticks.map((t) => t.price_usd)) * 0.9999
    : "auto";
  const domainMax = ticks.length > 1
    ? Math.max(...ticks.map((t) => t.price_usd)) * 1.0001
    : "auto";

  const hasOpenPos = !!portfolio?.open_position;
  const pnlIsPositive = (portfolio?.total_pnl_usd ?? 0) >= 0;

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-[#0a0c14] text-slate-200 font-sans p-4">
      <div className="max-w-7xl mx-auto space-y-5">

        {/* ── Header ─────────────────────────────────────────────────────── */}
        <header className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-2xl font-bold text-white flex items-center gap-2">
                <Activity className="text-blue-500 w-6 h-6" />
                Bitcoin Autonomous Trading & Prediction
              </h1>
              <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                PAPER TRADING (DUMMY MONEY)
              </span>
            </div>
            <p className="text-slate-500 text-sm mt-0.5">
              Phase 1: Performance Tracking & Predictive Loss Anticipation
            </p>
          </div>
          <div className="flex items-center gap-4 text-sm">
            {wsStatus === "live" ? (
              <span className="flex items-center gap-1.5 text-green-400">
                <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                <Wifi className="w-4 h-4" /> Live Market
              </span>
            ) : wsStatus === "connecting" ? (
              <span className="flex items-center gap-1.5 text-yellow-400">
                <span className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />
                Connecting…
              </span>
            ) : (
              <span className="flex items-center gap-1.5 text-red-400">
                <WifiOff className="w-4 h-4" /> Disconnected
              </span>
            )}

            {hasOpenPos && (
              <button
                onClick={handleEmergencyStop}
                disabled={isEmergencyStopping}
                className="px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-lg shadow-red-900/30"
              >
                <AlertTriangle className="w-3.5 h-3.5" />
                {isEmergencyStopping ? "Liquidating…" : "Emergency Withdraw"}
              </button>
            )}
          </div>
        </header>

        {/* ── Client Phase 1 Notice Banner ─────────────────────────────────── */}
        <div className="bg-blue-950/30 border border-blue-500/20 text-blue-300 px-4 py-3 rounded-xl flex items-center justify-between text-sm">
          <div className="flex items-center gap-3">
            <ShieldCheck className="w-5 h-5 text-blue-400 shrink-0" />
            <div>
              <span className="font-semibold text-white">Validation Gate Active:</span> The bot executes simulated trades using real-time Binance prices. It auto-withdraws when the model anticipates a loss or dynamic trailing stops trigger.
            </div>
          </div>
          <div className="hidden md:flex items-center gap-2 text-xs text-blue-400/80">
            <span>Stop Loss: 1.0%</span> • <span>Trailing: 0.5%</span> • <span>Reversal Exit: ON</span>
          </div>
        </div>

        {/* ── Paper Trading Portfolio Summary Cards ──────────────────────── */}
        <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
          <div className="bg-[#131722] rounded-xl p-4 border border-slate-800">
            <p className="text-slate-400 text-xs uppercase tracking-wider mb-1 flex items-center gap-1">
              <Wallet className="w-3.5 h-3.5 text-slate-500" /> Capital
            </p>
            <div className="text-xl font-mono font-bold text-white">
              ${portfolio?.balance_usd?.toLocaleString("en-US", { minimumFractionDigits: 2 }) ?? "1,000.00"}
            </div>
            <p className="text-slate-600 text-[11px] mt-0.5">Initial: ${portfolio?.initial_capital_usd ?? 1000}</p>
          </div>

          <div className="bg-[#131722] rounded-xl p-4 border border-slate-800">
            <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">Total Net P&L</p>
            <div className={`text-xl font-mono font-bold ${pnlIsPositive ? "text-emerald-400" : "text-rose-400"}`}>
              {pnlIsPositive ? "+" : ""}${portfolio?.total_pnl_usd?.toFixed(2) ?? "0.00"}
            </div>
            <p className={`text-[11px] mt-0.5 ${pnlIsPositive ? "text-emerald-500" : "text-rose-500"}`}>
              {pnlIsPositive ? "+" : ""}{portfolio?.total_return_pct?.toFixed(2) ?? "0.00"}%
            </p>
          </div>

          <div className="bg-[#131722] rounded-xl p-4 border border-slate-800">
            <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">Win Rate</p>
            <div className="text-xl font-mono font-bold text-white">
              {portfolio?.win_rate_pct?.toFixed(1) ?? "0.0"}%
            </div>
            <p className="text-slate-500 text-[11px] mt-0.5">
              {portfolio?.winning_trades ?? 0} wins / {portfolio?.total_trades ?? 0} total
            </p>
          </div>

          <div className="bg-[#131722] rounded-xl p-4 border border-slate-800">
            <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">Profit Factor</p>
            <div className="text-xl font-mono font-bold text-white">
              {portfolio?.profit_factor !== undefined ? (portfolio.profit_factor > 900 ? "∞" : portfolio.profit_factor.toFixed(2)) : "—"}
            </div>
            <p className="text-slate-600 text-[11px] mt-0.5">Target: &gt; 1.40</p>
          </div>

          <div className="bg-[#131722] rounded-xl p-4 border border-slate-800">
            <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">Max Drawdown</p>
            <div className="text-xl font-mono font-bold text-amber-400">
              {portfolio?.max_drawdown_pct?.toFixed(1) ?? "0.0"}%
            </div>
            <p className="text-slate-600 text-[11px] mt-0.5">Peak: ${portfolio?.peak_balance_usd?.toFixed(0) ?? "1000"}</p>
          </div>

          <div className="bg-[#131722] rounded-xl p-4 border border-slate-800">
            <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">Active Position</p>
            {hasOpenPos ? (
              <div className="flex items-center gap-1.5">
                <span className={`px-2 py-0.5 rounded text-xs font-bold ${portfolio?.open_position?.side === "BUY" ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"}`}>
                  {portfolio?.open_position?.side}
                </span>
                <span className="text-xs text-slate-300 font-mono">
                  ${portfolio?.open_position?.entry_price.toLocaleString()}
                </span>
              </div>
            ) : (
              <div className="text-sm font-semibold text-slate-500 flex items-center gap-1">
                <span>HOLDING CASH</span>
              </div>
            )}
            <p className="text-slate-600 text-[11px] mt-0.5">
              {hasOpenPos ? `Stop: $${portfolio?.open_position?.stop_loss_price?.toFixed(0)}` : "Ready to enter"}
            </p>
          </div>
        </div>

        {/* ── Live Alert Banner ───────────────────────────────────────────── */}
        {botAlert && (
          <div className={`rounded-xl p-4 border ${botAlert.action.includes("BUY") ? "bg-green-900/20 border-green-500/30" : botAlert.action.includes("CLOSE") ? "bg-purple-900/20 border-purple-500/30" : "bg-red-900/20 border-red-500/30"}`}>
            <div className="flex justify-between items-center mb-3">
              <h2 className="text-base font-bold flex items-center gap-2">
                🚨 BOT ACTION: <span className={botAlert.action.includes("BUY") ? "text-green-400" : botAlert.action.includes("CLOSE") ? "text-purple-400" : "text-red-400"}>{botAlert.action}</span>
              </h2>
              <span className="text-xs text-slate-400 font-mono">
                {new Date(botAlert.timestamp).toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" })}
              </span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <div>
                <p className="text-slate-400 text-xs uppercase mb-0.5">Trigger Price</p>
                <p className="text-base font-mono text-white">${botAlert.entry_price.toLocaleString("en-US", { minimumFractionDigits: 2 })}</p>
              </div>
              <div>
                <p className="text-slate-400 text-xs uppercase mb-0.5">Take Profit</p>
                <p className="text-base font-mono text-green-400">${botAlert.take_profit.toLocaleString("en-US", { minimumFractionDigits: 2 })}</p>
              </div>
              <div>
                <p className="text-slate-400 text-xs uppercase mb-0.5">Trailing Stop</p>
                <p className="text-base font-mono text-red-400">${botAlert.stop_loss.toLocaleString("en-US", { minimumFractionDigits: 2 })}</p>
              </div>
              <div>
                <p className="text-slate-400 text-xs uppercase mb-0.5">Model Confidence</p>
                <p className="text-base font-mono text-blue-400">{(botAlert.confidence * 100).toFixed(1)}%</p>
              </div>
            </div>
          </div>
        )}

        {/* ── Top row (Market Price & Next 1h Direction) ─────────────────── */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">

          {/* Live Price */}
          <div className="bg-[#131722] rounded-xl p-5 border border-slate-800">
            <div className="flex items-center justify-between mb-1">
              <p className="text-slate-400 text-xs uppercase tracking-widest">BTC / USD  •  Live Feed</p>
              <span className="text-xs px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 font-mono">
                1 USDT ≈ ${usdtRate.toFixed(4)}
              </span>
            </div>
            <div className={`text-4xl font-mono font-bold transition-colors duration-200 ${
              !priceChanged ? "text-white" : priceUp ? "text-green-400" : "text-red-400"
            }`}>
              {livePriceUsd !== null
                ? `$${livePriceUsd.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
                : "—"}
            </div>
            {livePriceUsdt !== null && (
              <p className="text-slate-500 text-xs mt-1">
                Raw Binance: ${livePriceUsdt.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} USDT
              </p>
            )}
            {priceChanged && (
              <p className={`text-sm mt-1 ${priceUp ? "text-green-400" : "text-red-400"}`}>
                {priceUp ? "▲" : "▼"} {Math.abs((livePriceUsd! - prevPriceUsd!)).toFixed(2)} USD
              </p>
            )}
          </div>

          {/* 1h Prediction */}
          <div className="bg-[#131722] rounded-xl p-5 border border-slate-800">
            <p className="text-slate-400 text-xs uppercase tracking-widest mb-1">Next 1h AI Prediction</p>
            <div className="text-3xl font-bold flex items-center gap-2">
              {prediction ? (
                isUp ? <span className="text-green-400 flex items-center gap-2"><TrendingUp /> UP (Bullish)</span>
                : isDown ? <span className="text-red-400 flex items-center gap-2"><TrendingDown /> DOWN (Bearish)</span>
                : <span className="text-slate-300 flex items-center gap-2"><Minus /> NEUTRAL (Range)</span>
              ) : <span className="text-slate-600 text-lg">Loading…</span>}
            </div>
            <div className="flex justify-between items-end mt-2">
              <p className="text-slate-500 text-xs">Model: {prediction?.model_version ?? "—"}</p>
              {prediction?.timestamp && (
                <p className="text-slate-500 text-xs font-mono">
                  {new Date(prediction.timestamp).toLocaleString("en-US", { 
                    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" 
                  })}
                </p>
              )}
            </div>
          </div>

          {/* Confidence */}
          <div className="bg-[#131722] rounded-xl p-5 border border-slate-800">
            <p className="text-slate-400 text-xs uppercase tracking-widest mb-1">Model Confidence</p>
            <div className="text-4xl font-mono font-bold text-white">
              {prediction ? `${(prediction.confidence * 100).toFixed(1)}%` : "—"}
            </div>
            <div className="w-full bg-slate-800 h-1.5 mt-3 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  isUp ? "bg-green-500" : isDown ? "bg-red-500" : "bg-slate-400"
                }`}
                style={{ width: `${(prediction?.confidence ?? 0) * 100}%` }}
              />
            </div>
            <p className="text-slate-600 text-xs mt-2">
              Auto-Trade Threshold: &ge; 60%  •  Loss-Exit Threshold: &ge; 50%
            </p>
          </div>
        </div>

        {/* ── Main content: Chart & Probabilities ────────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">

          {/* Live price chart */}
          <div className="lg:col-span-2 bg-[#131722] rounded-xl p-5 border border-slate-800 flex flex-col">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-white uppercase tracking-widest">
                Real-Time Price Stream <span className="text-slate-600 font-normal normal-case">({ticks.length} ticks)</span>
              </h3>
              <span className="text-xs text-slate-500">aggTrade stream • sub-second</span>
            </div>
            <div className="flex-1 min-h-[300px]">
              {ticks.length < 2 ? (
                <div className="flex items-center justify-center h-full text-slate-600 text-sm">
                  Waiting for trade data from Binance WebSocket…
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={ticks} margin={{ top: 4, right: 4, left: 16, bottom: 4 }}>
                    <defs>
                      <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.25} />
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <XAxis
                      dataKey="time"
                      tick={{ fill: "#475569", fontSize: 11 }}
                      stroke="#1e293b"
                      interval="preserveStartEnd"
                    />
                    <YAxis
                      domain={[domainMin, domainMax]}
                      tick={{ fill: "#475569", fontSize: 11 }}
                      stroke="#1e293b"
                      tickFormatter={(v) => `$${v.toLocaleString()}`}
                      width={90}
                    />
                    <Tooltip
                      contentStyle={{ backgroundColor: "#0a0c14", borderColor: "#1e293b", color: "#f8fafc" }}
                      formatter={(v: any) => [`$${Number(v ?? 0).toFixed(2)}`, "Price"]}
                      labelFormatter={(l) => `Time: ${l}`}
                    />
                    <Area
                      type="monotone"
                      dataKey="price_usd"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      fill="url(#priceGrad)"
                      dot={false}
                      isAnimationActive={false}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>

          {/* Probabilities */}
          <div className="bg-[#131722] rounded-xl p-5 border border-slate-800 flex flex-col gap-6">
            <div>
              <h3 className="text-sm font-semibold text-white uppercase tracking-widest mb-4">
                Directional Probabilities
              </h3>
              <div className="space-y-4">
                {[
                  { label: "UP (>+0.15%)", value: prediction?.probability_up ?? 0, color: "bg-green-500", text: "text-green-400" },
                  { label: "NEUTRAL", value: prediction?.probability_neutral ?? 0, color: "bg-slate-400", text: "text-slate-300" },
                  { label: "DOWN (<-0.15%)", value: prediction?.probability_down ?? 0, color: "bg-red-500", text: "text-red-400" },
                ].map(({ label, value, color, text }) => (
                  <div key={label}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className={`font-medium ${text}`}>{label}</span>
                      <span className="font-mono text-white">{(value * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-slate-800 h-2 rounded-full">
                      <div className={`${color} h-2 rounded-full transition-all duration-500`} style={{ width: `${value * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="border-t border-slate-800 pt-4">
              <h3 className="text-xs text-slate-500 uppercase tracking-widest mb-3">Recent Ticks</h3>
              <div className="space-y-2 font-mono text-xs">
                {ticks.slice(-5).reverse().map((t, i) => (
                  <div key={i} className="flex justify-between text-slate-400">
                    <span className="text-slate-600">{t.time}</span>
                    <div className="text-right">
                      <span className="text-white block">${t.price_usd.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                      <span className="text-slate-600 text-xs">{t.price_usdt.toFixed(2)} USDT</span>
                    </div>
                  </div>
                ))}
                {ticks.length === 0 && <p className="text-slate-700">Waiting for stream…</p>}
              </div>
            </div>
          </div>
        </div>

        {/* ── Closed Trades History (Validation Tracking) ────────────────── */}
        <div className="bg-[#131722] rounded-xl p-5 border border-slate-800">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-semibold text-white uppercase tracking-widest">
                Paper Trade Execution History ({portfolio?.recent_trades?.length ?? 0} trades logged)
              </h3>
              <p className="text-xs text-slate-500 mt-0.5">
                Tracks each autonomous trade with entry, exit, P&amp;L, and withdrawal trigger
              </p>
            </div>
            <span className="text-xs text-slate-500">Auto-logged to SQLite</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-slate-900/60 text-slate-400 uppercase border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3">Side</th>
                  <th className="py-2.5 px-3">Entry Price</th>
                  <th className="py-2.5 px-3">Exit Price</th>
                  <th className="py-2.5 px-3">Size (BTC)</th>
                  <th className="py-2.5 px-3">P&amp;L (USD)</th>
                  <th className="py-2.5 px-3">P&amp;L (%)</th>
                  <th className="py-2.5 px-3">Withdrawal Reason</th>
                  <th className="py-2.5 px-3">Closed Time</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {(!portfolio?.recent_trades || portfolio.recent_trades.length === 0) ? (
                  <tr>
                    <td colSpan={8} className="py-6 text-center text-slate-600 font-sans">
                      No closed trades yet. The bot will automatically execute and log paper trades when model confidence exceeds 60%.
                    </td>
                  </tr>
                ) : (
                  portfolio.recent_trades.map((t, idx) => {
                    const isWin = (t.pnl_usd ?? 0) > 0;
                    const reason = t.reason || t.exit_reason || "signal";
                    return (
                      <tr key={idx} className="hover:bg-slate-800/30 transition-colors">
                        <td className="py-2.5 px-3">
                          <span className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                            t.side === "BUY" ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
                          }`}>
                            {t.side}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-slate-300">${t.entry_price?.toLocaleString("en-US", { minimumFractionDigits: 2 })}</td>
                        <td className="py-2.5 px-3 text-slate-300">${t.exit_price?.toLocaleString("en-US", { minimumFractionDigits: 2 })}</td>
                        <td className="py-2.5 px-3 text-slate-400">{t.quantity_btc?.toFixed(6)}</td>
                        <td className={`py-2.5 px-3 font-bold ${isWin ? "text-emerald-400" : "text-rose-400"}`}>
                          {isWin ? "+" : ""}${t.pnl_usd?.toFixed(2)}
                        </td>
                        <td className={`py-2.5 px-3 font-bold ${isWin ? "text-emerald-400" : "text-rose-400"}`}>
                          {isWin ? "+" : ""}{t.pnl_pct?.toFixed(2)}%
                        </td>
                        <td className="py-2.5 px-3">
                          <span className={`px-2 py-0.5 rounded text-[10px] uppercase font-semibold ${
                            reason === "predictive_reversal"
                              ? "bg-purple-500/20 text-purple-300 border border-purple-500/30"
                              : reason === "trailing_stop"
                              ? "bg-blue-500/20 text-blue-300 border border-blue-500/30"
                              : reason === "emergency_stop"
                              ? "bg-red-500/20 text-red-300 border border-red-500/30"
                              : "bg-slate-700/40 text-slate-300"
                          }`}>
                            {reason.replace("_", " ")}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-slate-500 text-[11px]">
                          {t.closed_at ? new Date(t.closed_at).toLocaleTimeString() : "—"}
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </div>
  );
}
